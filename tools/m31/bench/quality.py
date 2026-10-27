#!/usr/bin/env python3
"""Run the historical GSM8K100 and tools23 recipes against an OpenAI chat API.

Tools are scored as data. This program never executes a generated tool call.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

from tool_fixtures import CASES, FOLLOW, TOOLS


def fingerprint(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def number(value):
    try:
        return float(value.replace(",", "").replace("$", "").strip().rstrip("."))
    except ValueError:
        return None


def uint64_seed(value):
    try:
        seed = int(value, 10)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("seed must be an unsigned 64-bit integer") from None
    if seed < 0 or seed > (1 << 64) - 1:
        raise argparse.ArgumentTypeError("seed must be between 0 and 18446744073709551615")
    return seed


def inspect_choice(choice):
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    calls = msg.get("tool_calls") or []
    identifiers = [call.get("id") for call in calls]
    return {
        "finish": choice.get("finish_reason"),
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
        "marker_leak": bool(re.search(r"</?(?:mm:)?think>|<tool_call>|<arg_key>|<\|(user|assistant|observation|system)\|>|\]<\]minimax\[>\[|\]~b\]|\[e~\[|\]!p~\[|\]!d~\[", content)),
        "invalid_tool_ids": any(not isinstance(i, str) or not i for i in identifiers) or len(identifiers) != len(set(identifiers)),
    }


class Evaluation:
    def __init__(self, args, results, raw):
        self.args, self.results, self.raw = args, results, raw
        self.records = []

    def chat(self, messages, tools=None):
        # Historical quality used model-default thinking, not reasoning_effort=low.
        body = {"model": self.args.model, "messages": messages, "temperature": 1.0,
                "top_p": .95, "top_k": 40, "min_p": .05, "max_tokens": 6000, "stream": False}
        if tools:
            body["tools"] = tools
        if getattr(self.args, "seed", None) is not None:
            body["seed"] = self.args.seed
        request = urllib.request.Request(self.args.url.rstrip("/") + "/v1/chat/completions",
            json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
        if self.args.api_key:
            request.add_header("Authorization", "Bearer " + self.args.api_key)
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                data = json.load(response)
        except BaseException as exc:
            self.raw.write(json.dumps({"request": body, "error": f"{type(exc).__name__}: {exc}",
                "seconds": time.monotonic() - start}, ensure_ascii=False) + "\n")
            self.raw.flush()
            raise
        duration = time.monotonic() - start
        self.raw.write(json.dumps({"request": body, "response": data, "seconds": duration}, ensure_ascii=False) + "\n")
        self.raw.flush()
        return data["choices"][0], data.get("usage", {}), duration

    def record(self, choice, usage, duration, **values):
        row = {**values, **inspect_choice(choice), "usage": usage, "seconds": duration}
        self.records.append(row)
        self.results.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.results.flush()
        print(json.dumps({k: row[k] for k in ("kind", "id", "ok", "finish")}), flush=True)

    def gsm8k(self):
        rows = [json.loads(line) for line in self.args.dataset.read_text().splitlines() if line.strip()][:100]
        if len(rows) != 100:
            raise ValueError("GSM8K100 requires at least 100 ordered test rows")
        for i, row in enumerate(rows):
            gold = number(row["answer"].split("####")[-1])
            prompt = row["question"] + "\n\nSolve step by step, then give the final answer on the last line as: Answer: <number>"
            choice, usage, duration = self.chat([{"role": "user", "content": prompt}])
            text = choice["message"].get("content") or ""
            matches = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
            if not matches:
                matches = re.findall(r"(-?[\d,]*\.?\d+)", text)
            pred = number(matches[-1]) if matches else None
            good = pred is not None and gold is not None and abs(pred - gold) < 1e-6
            self.record(choice, usage, duration, kind="gsm8k", id=i, ok=good, pred=pred, gold=gold)

    def tools23(self):
        for i, (query, wanted, check) in enumerate(CASES):
            choice, usage, duration = self.chat([{"role": "user", "content": query}], TOOLS)
            calls = choice["message"].get("tool_calls") or []
            got, arguments, error = None, None, None
            if calls:
                got = calls[0]["function"]["name"]
                try:
                    arguments = json.loads(calls[0]["function"]["arguments"])
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
            try:
                good = not calls if wanted is None else got == wanted and arguments is not None and bool(check(arguments))
            except (ValueError, KeyError, TypeError) as exc:
                good, error = False, str(exc)
            self.record(choice, usage, duration, kind="tool", id=i, ok=good, want=wanted, got=got, args=arguments, error=error)
        for i, (query, name, arguments, result, required) in enumerate(FOLLOW):
            messages = [{"role": "user", "content": query}, {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(result, ensure_ascii=False)}]
            choice, usage, duration = self.chat(messages, TOOLS)
            text = (choice["message"].get("content") or "").lower().replace("\u2212", "-")
            good = all(part.lower() in text for part in required) and not choice["message"].get("tool_calls")
            self.record(choice, usage, duration, kind="tool_follow", id=i, ok=good)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="minimax-m31")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--suite", choices=("all", "gsm8k", "tools23"), default="all")
    parser.add_argument("--dataset", type=Path, help="OpenAI GSM8K main/test.jsonl, in original order")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--seed", type=uint64_seed, default=None,
                        help="optional uint64 request seed; omitted by default")
    args = parser.parse_args()
    if args.suite != "tools23" and not args.dataset:
        parser.error("--dataset is required for GSM8K")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {"format": "m31-quality-v1", "suite": args.suite, "complete": False, "tool_execution": False,
        "sampling": {"temperature": 1, "top_p": .95, "top_k": 40, "min_p": .05, "max_tokens": 6000, "thinking": "model default"},
        "client": fingerprint(__file__), "fixtures": fingerprint(Path(__file__).with_name("tool_fixtures.py"))}
    if args.seed is not None:
        summary["sampling"]["seed"] = args.seed
    if args.dataset:
        summary["dataset"] = fingerprint(args.dataset)
    started = time.monotonic()
    with (args.out / "results.jsonl").open("x") as results, (args.out / "raw.jsonl").open("x") as raw:
        evaluation = Evaluation(args, results, raw)
        try:
            if args.suite != "tools23":
                evaluation.gsm8k()
            if args.suite != "gsm8k":
                evaluation.tools23()
            summary["complete"] = True
        except BaseException as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rows = evaluation.records
            gsm = [r for r in rows if r["kind"] == "gsm8k"]
            tools = [r for r in rows if r["kind"].startswith("tool")]
            summary.update(gsm8k={"correct": sum(r["ok"] for r in gsm), "total": len(gsm)},
                           tools23={"correct": sum(r["ok"] for r in tools), "total": len(tools)},
                           length_stops=sum(r["finish"] == "length" for r in rows),
                           marker_leaks=sum(r["marker_leak"] for r in rows),
                           invalid_tool_ids=sum(r["invalid_tool_ids"] for r in rows),
                           wall_seconds=time.monotonic() - started)
            summary["pass"] = (summary["complete"] and (args.suite == "tools23" or len(gsm) == 100 and sum(r["ok"] for r in gsm) >= 95)
                and (args.suite == "gsm8k" or len(tools) == 23 and sum(r["ok"] for r in tools) >= 19)
                and not summary["length_stops"] and not summary["marker_leaks"] and not summary["invalid_tool_ids"])
            (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
