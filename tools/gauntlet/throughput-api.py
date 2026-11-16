#!/usr/bin/env python3
"""throughput-api.py - prefill + decode speeds of a remote OpenAI-compatible API, sampled over the wire.

tools/throughput.py reads the engine's own timings (prompt_n/prompt_ms, predicted_n/predicted_ms); an API returns none,
so the only honest instrument is the stream: time-to-first-content-token is the prefill, the gap between the first and
last content delta is the decode.

  throughput-api.py <url> --model <id> [--prompt 8000] [--decode 256] [--key-env OPENROUTER_API_KEY]

1. prefill: a fresh ~--prompt-token user message from a code corpus, max_tokens 1 -> prompt_tokens / TTFT.
2. decode:  the same conversation + "continue" asking for --decode tokens -> (n - 1) / (t_last - t_first).
   Thinking is switched off for the timing run so the number is decode of the answer, not of a chain of thought
   (`reasoning_effort: none`; the field is dropped and retried if the provider rejects it).
All numbers are e2e (network included), so they are a lower bound on the provider's engine speed.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
from turnbench import corpus  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def stream(url, model, msgs, max_tokens, key, thinking_off=True):
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if thinking_off:
        body["reasoning_effort"] = "none"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key

    def attempt(b):
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(), headers=headers)
        t0 = time.time()
        t_event = t_first = None
        n = 0
        usage = {}
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if j.get("usage"):
                    usage = j["usage"]
                if t_event is None and j.get("choices"):
                    t_event = time.time()  # first event = prefill done
                for ch in j.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if t_first is None:
                            t_first = time.time()
                        n += 1
        return {"ttft": (t_event - t0) if t_event else None,
                "t_first": (t_first - t0) if t_first else None,
                "wall": time.time() - t0, "deltas": n, "usage": usage}

    try:
        return attempt(body)
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "replace")
        if thinking_off and e.code == 400:
            body.pop("reasoning_effort", None)
            r = attempt(body)
            r["note"] = "reasoning_effort rejected, retried without it: " + detail
            return r
        raise


def main():
    url = sys.argv[1]
    model = arg("--model", os.environ.get("TASKEVAL_MODEL", "x"))
    n_prompt, n_dec = int(arg("--prompt", 8000)), int(arg("--decode", 256))
    key = os.environ.get(arg("--key-env", "OPENROUTER_API_KEY"), "")
    out_json = arg("--json", "")

    text = corpus()
    cur = 40000 + int(time.time()) % 20000
    take = lambda n: text[cur:cur + int(n * 3.3)]
    msgs = [{"role": "user", "content": "Read this code carefully:\n" + take(n_prompt)}]

    a = stream(url, model, msgs, 1, key, thinking_off=True)
    pt = a["usage"].get("prompt_tokens") or 0
    pre = pt / a["ttft"] if a["ttft"] else 0

    msgs.append({"role": "assistant", "content": "Noted."})
    msgs.append({"role": "user", "content": "Continue with a detailed explanation of the architecture above."})
    b = stream(url, model, msgs, n_dec, key, thinking_off=True)
    ct = b["usage"].get("completion_tokens") or b["deltas"]
    span = (b["wall"] - b["t_first"]) if b.get("t_first") else 0
    dec = (ct - 1) / span if span > 0 else 0

    row = {"model": model, "prompt_target": n_prompt, "prompt_tokens": pt, "prefill_ttft_s": a["ttft"],
           "prefill_tok_s": pre, "decode_target": n_dec, "decode_tokens": ct, "decode_span_s": span,
           "decode_tok_s": dec, "decode_wall_s": b["wall"], "note": b.get("note", "")}
    print(json.dumps(row, indent=1))
    if out_json:
        json.dump(row, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
