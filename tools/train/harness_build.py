#!/usr/bin/env python3
"""Build a QuetzaCodetl HARNESS SFT set for QFN from our own chat corpus, rendered exactly as Xolotl serve renders a
real Quetza request.

usage (Xolotl's venv: serve + tools on the path):
  ~/AI/Xolotl/.venv/bin/python harness_build.py --chats chats.jsonl --req req-001.json --out DIR
        [--max-conv-tokens 2048] [--producers REGEX] [--limit N] [--sources quetza,quetza-conductor]

Inputs
  chats.jsonl  your own agent transcripts, one conversation per line (messages + tool calls, format-neutral). It has messages and tool calls but NOT the
               system prompt or the tool schemas - Claude-Code-style logs never store those.
  req-001.json one real request QuetzaCodetl sent to a local model (harness_capture_server.py): the CURRENT system
               prompt (~33K chars) and the 14 tool schemas (~42K chars). Every example uses them, so the rendered
               prefix (system + tools) is IDENTICAL across examples - the trainer computes it once.

Rendering = serve's own path: frontend.anthropic_to_messages(req) -> ChatTemplate(<pack tokenizer>/chat_template.jinja)
.render(...) -> strata_tokenizer (the ids were checked equal to llama-tokenize on the pack). The QFN template keeps
every turn's reasoning in the history (preserve_thinking defaults to true), so one render of a window supervises all
of its assistant turns.

Loss mask: assistant turns whose producer matches --producers (strong models; QFN's own turns are excluded) and whose
tool calls all name a tool Quetza still offers. Span = from the end of the generation prompt
(`<|im_start|>assistant\\n<think>\\n`) through the turn's `<|im_end|>`.

Windows: the first user message (environment + task) is always kept; the rest is cut into consecutive segments that
start at an assistant turn and fit --max-conv-tokens together with it. (QSA layer 47 is exact-dense only while the
whole sequence fits the indexer's 2048 cells; longer sequences need the engine's selection replayed - see LORA_PLAN.)

Outputs in DIR: prefix.json (prefix token ids + text hash), windows.jsonl (one window per line: conv id, token ids
AFTER the prefix, loss mask), stats.json.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

XOL = Path(__file__).resolve().parents[2]   # the Xolotl checkout
sys.path.insert(0, str(XOL / "tools"))
sys.path.insert(0, str(XOL))
from serve.frontend import ChatTemplate, anthropic_to_messages  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

TOKDIR = Path(os.environ.get("QFN_PACK", "pack")) / "tokenizer"   # QFN_PACK: the QFN pack directory
DEFAULT_PRODUCERS = r"deepseek|claude-opus|claude-fable|kimi|glm-5|qwen3\.7|qwen4-max"


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
    """corpus message -> Anthropic message (None when nothing renderable is left)."""
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
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + m["content"], "_src": out[-1]["_src"] + m["_src"]}
        else:
            out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chats", required=True)
    ap.add_argument("--req", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-conv-tokens", type=int, default=2048)
    ap.add_argument("--producers", default=DEFAULT_PRODUCERS)
    ap.add_argument("--sources", default="quetza,quetza-conductor,claude-code,claude-data")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    req0 = json.loads(Path(a.req).read_text())
    tool_names = {t["name"] for t in req0.get("tools") or []}
    tool_params = {t["name"]: set(((t.get("input_schema") or {}).get("properties") or {}).keys())
                   for t in req0.get("tools") or []}
    tpl = ChatTemplate(TOKDIR / "chat_template.jinja")
    tok = load_tokenizer(TOKDIR)
    prod = re.compile(a.producers)
    local_q = re.compile(r"strata|xq|gsq|xolotl|abl|gguf|<synthetic>", re.I)
    sources = set(a.sources.split(","))

    def render(msgs, gen):
        req = {k: v for k, v in req0.items() if k not in ("messages",)}
        req["messages"] = [{"role": m["role"], "content": m["content"]} for m in msgs]
        messages, tools, kw = anthropic_to_messages(req)
        # serve's cache-point marker (a random token per request at each cache_control point): encode_prompt strips it
        # out of the prompt before tokenizing, so the text the model sees has none
        marker = kw.pop("_strata_cache_marker", None)
        text = tpl.render(messages, tools=tools, add_generation_prompt=gen, **kw)
        return text.replace(marker, "") if marker else text

    # the shared prefix: everything before the first user turn
    probe = render([{"role": "user", "content": [{"type": "text", "text": "x"}]}], False)
    cut = probe.find("<|im_start|>user")
    prefix_text = probe[:cut]
    prefix_ids = tok.encode(prefix_text, parse_special=True)
    (out / "prefix.json").write_text(json.dumps({"tokens": prefix_ids, "n": len(prefix_ids),
                                                "sha256": hashlib.sha256(prefix_text.encode()).hexdigest(),
                                                "tools": sorted(tool_names)}))
    print(f"[build] prefix: {len(prefix_text)} chars -> {len(prefix_ids)} tokens; tools {sorted(tool_names)}", flush=True)

    st = {"convs_seen": 0, "convs_used": 0, "windows": 0, "loss_tokens": 0, "tokens": 0, "turns_trained": 0,
          "turns_masked_tool": 0, "turns_masked_producer": 0, "split_mismatch": 0, "prefix_mismatch": 0,
          "by_producer": {}}
    # --chats may name several builds (comma-separated): one conversation per id, the copy with the most messages
    # (a later build can miss histories that were cleaned up on disk since an earlier one)
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
    if True:
        for r in sorted(best.values(), key=lambda x: x.get("started") or ""):
            if r.get("source") not in sources:
                continue
            ms = r.get("messages") or []
            if not ms or (ms[0].get("meta") or {}).get("isSidechain"):
                continue
            st["convs_seen"] += 1
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
                # strong API producers only: our own local quants (DS4.1 XQ/GSQ, strata builds) are not teachers
                if not any(prod.search(x) and not local_q.search(x) for x in models):
                    return "producer"
                calls = [b for b in am["content"] if b["type"] == "tool_use"]
                if any(c["name"] not in tool_names for c in calls):
                    return "tool"
                # QuetzaCodetl is a Claude Code fork: same tools, but an argument the CURRENT schema lacks would teach
                # a call Quetza rejects (99 % of Claude Code calls fit, 21 Nov)
                if any(set(c["input"]) - tool_params[c["name"]] for c in calls):
                    return "tool"
                return None

            first, rest = conv[0], conv[1:]
            first_ids = len(tok.encode(render([first], False)[cut:], parse_special=True))
            if first_ids > 1024:
                # Claude Code's first message carries 14-40K tokens of harness attachments (CLAUDE.md, memory, hook
                # output, skill lists); keep only what was typed (text blocks that did not come from an attachment)
                typed = []
                for src in first["_src"]:
                    for b in src.get("content") or []:
                        if b.get("type") == "text" and b.get("text"):
                            typed.append({"type": "text", "text": b["text"]})
                if typed:
                    first = {"role": "user", "content": typed, "_src": first["_src"]}
                    first_ids = len(tok.encode(render([first], False)[cut:], parse_special=True))
                    st["first_stripped"] = st.get("first_stripped", 0) + 1
            budget = a.max_conv_tokens - first_ids
            if budget < 256:
                continue
            # segments that start at an assistant turn
            i, used = 0, False
            while i < len(rest):
                if rest[i]["role"] != "assistant":
                    i += 1
                    continue
                seg, j, n_est = [], i, 0
                while j < len(rest):
                    est = sum(len(json.dumps(b)) for b in rest[j]["content"]) // 3 + 8   # cheap chars/3 estimate
                    if seg and n_est + est > budget:
                        break
                    seg.append(rest[j])
                    n_est += est
                    j += 1
                i = j
                msgs = [first] + seg
                full = render(msgs, False)
                if not full.startswith(prefix_text):
                    st["prefix_mismatch"] += 1
                    continue
                spans = []
                for k in range(1, len(msgs)):
                    if msgs[k]["role"] != "assistant":
                        continue
                    why = trainable(msgs[k])
                    if why:
                        st["turns_masked_" + why] += 1
                        continue
                    g = render(msgs[:k], True)
                    e = render(msgs[:k + 1], False)
                    if not full.startswith(g) or not full.startswith(e) or len(e) <= len(g):
                        st["split_mismatch"] += 1
                        if st["split_mismatch"] <= 4:
                            kind = "gen" if not full.startswith(g) else "end" if not full.startswith(e) else "empty"
                            a_, b_ = (g, full) if kind == "gen" else (e, full) if kind == "end" else (e, g)
                            i_ = next((i for i, (x, y) in enumerate(zip(a_, b_)) if x != y), min(len(a_), len(b_)))
                            with open(out / "mismatch.txt", "a") as mf:
                                mf.write(f"===== {kind} turn {k}/{len(msgs)} blocks {[bl['type'] for bl in msgs[k]['content']]}\n"
                                         f"{a_[max(0, i_ - 400):i_ + 150]!r}\n---\n{b_[max(0, i_ - 400):i_ + 150]!r}\n")
                        continue
                    end = e.find("<|im_end|>", len(g))
                    end = end + len("<|im_end|>") if end >= 0 else len(e)
                    spans.append((len(g), end, next(iter({s.get("model") for s in msgs[k]["_src"]}))))
                if not spans:
                    continue
                # tokenize the window ONCE (pieces would let BPE merge across a span edge, e.g. "<think>\n" + "\n...")
                # and give each token to the loss by its byte offset: a token is a loss token when it STARTS inside a
                # span. The ids are the ones the served model would see for this text.
                body = full[len(prefix_text):]
                ids = tok.encode(body, parse_special=True)
                enc = body.encode("utf-8")
                base = len(prefix_text)
                bspans = [(len(full[base:s0].encode("utf-8")), len(full[base:s1].encode("utf-8"))) for s0, s1, _ in spans]
                mask, off = [], 0
                for t in ids:
                    tb = tok.token_bytes(t)
                    mask.append(1 if any(b0 <= off < b1 for b0, b1 in bspans) else 0)
                    off += len(tb)
                if off != len(enc):
                    st["split_mismatch"] += 1
                    continue
                if len(ids) > a.max_conv_tokens * 2:      # the char estimate was far off: drop rather than overrun
                    continue
                for _, _, mdl in spans:
                    st["by_producer"][mdl] = st["by_producer"].get(mdl, 0) + 1
                st["turns_trained"] += len(spans)
                st["windows"] += 1
                st["tokens"] += len(ids)
                st["loss_tokens"] += sum(mask)
                used = True
                wf.write(json.dumps({"conv": r["id"], "started": r.get("started"), "n": len(ids),
                                     "tokens": ids, "mask": mask}) + "\n")
            st["convs_used"] += used
            if a.limit and st["windows"] >= a.limit:
                break
            if st["convs_seen"] % 25 == 0:
                print(f"[build] {st['convs_seen']} convs, {st['windows']} windows, {st['loss_tokens']} loss tokens",
                      flush=True)
    wf.close()
    (out / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
