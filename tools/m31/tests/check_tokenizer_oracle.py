#!/usr/bin/env python3
"""Compare the native Python adapter with a CPU vocabulary-only llama.cpp oracle.

The oracle is optional test tooling, never a server runtime dependency. Build
tools/m31/tests/oracle against the documented reference and pass its executable.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools/m31/serve")]
from gguf_reader import GGUFFile  # noqa: E402
from export_tokenizer import export  # noqa: E402
from m31_tok import M31Tokenizer  # noqa: E402
from serve_m31 import M31Template, M31Service, M31Detokenizer  # noqa: E402
from m31_parser import M31OutputParser, NS, CALL_START, CALL_END, THINK_START, THINK_END  # noqa: E402
from serve import server as S  # noqa: E402


def check_runtime_token_ids(tokenizer, template, tools):
    """Exercise the real vocabulary through Service.run without model inference."""
    def run(ids):
        engine = S.MockEngine(tokenizer, "")
        engine.script = ids + tokenizer.encode("[e~[", parse_special=True)
        engine.scripts = [engine.script]
        service = M31Service(engine, tokenizer, template)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(S, "OutputParser", M31OutputParser), \
                mock.patch.object(S, "Detokenizer", M31Detokenizer):
            items = list(service.run([1], True, tools, 2048, {"temperature": 0}, threading.Event()))
        return [value for kind, value in items if kind == "event"]
    def plain(text):
        return tokenizer.encode(text, parse_special=False, plain=[(0, len(text))])
    thinking = THINK_START + "why😀" + THINK_END + "answer東京"
    control_ids = tokenizer.encode(thinking, parse_special=True)
    if control_ids != tokenizer.encode(thinking, parse_special=False) or control_ids == plain(thinking):
        raise AssertionError("USER_DEFINED markers or explicit plain spans lost token identity")
    events = run(control_ids)
    if "".join(event.text for event in events if event.kind == "reasoning") != "why😀":
        raise AssertionError("actual thinking control IDs were not recognized")
    events = run(plain(thinking))
    if any(event.kind != "content" for event in events) or "".join(event.text for event in events) != thinking:
        raise AssertionError("ordinary thinking marker spelling became a control")
    prefix = CALL_START + NS + '<invoke name="lookup">' + NS + "<query>"
    literal = "Москва😀 " + NS + "</query> " + THINK_END
    suffix = NS + "</query>" + NS + "<values>" + NS + "</values>" + NS + "</invoke>" + CALL_END
    events = run(tokenizer.encode(prefix, parse_special=True) + plain(literal) + tokenizer.encode(suffix, parse_special=True))
    calls = [event.call for event in events if event.kind == "tool_call"]
    if len(calls) != 1 or calls[0].arguments != {"query": literal, "values": []}:
        raise AssertionError("ordinary namespace spelling closed a tool string")
    ordinary_tool = prefix + literal + suffix
    events = run(plain(ordinary_tool))
    if any(event.kind != "content" for event in events) or "".join(event.text for event in events) != ordinary_tool:
        raise AssertionError("ordinary namespace spelling became a tool call")
    return 4


def check(model, oracle):
    metadata = GGUFFile(Path(model)).metadata
    tokenizer = M31Tokenizer(metadata["tokenizer.ggml.tokens"], metadata["tokenizer.ggml.merges"],
                             metadata["tokenizer.ggml.token_type"])
    process = subprocess.Popen([str(Path(oracle).resolve()), str(Path(model).resolve())], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
    def read():
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("vocabulary oracle ended unexpectedly")
        value = json.loads(line)
        if "error" in value:
            raise RuntimeError(value["error"])
        return value
    def call(request):
        process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        process.stdin.flush()
        return read()
    counts = {"tokenize": 0, "chat": 0, "parse": 0, "pieces": 0, "runtime_token_ids": 0}
    try:
        ready = read()
        if ready["vocab"] != len(tokenizer.tokens):
            raise AssertionError("oracle vocabulary size differs")
        corpus = ["", "Hello, world!", "  lead  tail  ", "\r\n\tline\n", "a\x00b\x01", "1234567890123",
                  "DON'T I'll we've", "MixedCASE_and-dashes", "你好，世界", "abc漢字def", "Москва & Санкт-Петербург",
                  "مرحبا بالعالم", "नमस्ते", "e\u0301", "é", "가가", "😀🚀🇺🇸", "\u2028\u2029", "x" * 5000,
                  "]~!b[]~b]ai\n", "[e~[", NS, "<mm:think>", "</mm:think>", NS + "<tool_call>"]
        corpus += [chr(cp) for cp in range(256)]
        for text in corpus:
            for parse_special in (False, True):
                wanted = call({"op": "tokenize", "content": text, "parse_special": parse_special})["tokens"]
                actual = tokenizer.encode(text, parse_special=parse_special)
                if actual != wanted:
                    raise AssertionError(f"tokenization differs: {text!r}, special={parse_special}: {actual} != {wanted}")
                counts["tokenize"] += 1
        for token in sorted(set(ready.get("eog", [])) | set(tokenizer.special_tokens.values())):
            wanted = bytes(call({"op": "piece", "token": token})["bytes"])
            if tokenizer.token_bytes(token) != wanted:
                raise AssertionError(f"special token piece differs: {token}")
            counts["pieces"] += 1
        tools = [{"type": "function", "function": {"name": "lookup", "description": "Unicode lookup",
                  "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                 "values": {"type": "array", "items": {"type": "integer"}}},
                                 "required": ["query", "values"]}}}]
        cases = [([{"role": "user", "content": "hi"}], [], {}),
                 ([{"role": "system", "content": "Be concise."}, {"role": "user", "content": "東京"}], [], {}),
                 ([{"role": "user", "content": "hi"}], [], {"reasoning_effort": "low"}),
                 ([{"role": "user", "content": "hi"}], [], {"enable_thinking": False}),
                 ([{"role": "user", "content": "search"}], tools, {}),
                 ([{"role": "user", "content": "search"}], [{"type": "function", "function": {
                     "parameters": {"type": "object", "properties": {}}, "strict": True, "name": "lookup"}}], {}),
                 ([{"role": "user", "content": "search"}, {"role": "assistant", "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments":
                                   {"query": "Москва", "values": [1, 2]}}}]},
                   {"role": "tool", "content": "result"}], tools, {})]
        with tempfile.TemporaryDirectory(prefix="m31-vocab-check-") as directory:
            export(model, directory)
            template = M31Template(Path(directory) / "chat_template.jinja")
            counts["runtime_token_ids"] = check_runtime_token_ids(tokenizer, template, tools)
            for messages, toolset, kwargs in cases:
                request = {"op": "chat", "messages": messages, "tools": toolset,
                           "enable_thinking": kwargs.get("enable_thinking", True), "chat_template_kwargs": kwargs}
                wanted = call(request)
                prompt = template.render(messages, tools=toolset, **kwargs)
                if prompt != wanted["prompt"]:
                    at = next((i for i, (a, b) in enumerate(zip(prompt, wanted["prompt"])) if a != b),
                              min(len(prompt), len(wanted["prompt"])))
                    raise AssertionError(f"chat template differs at {at}: {prompt[at:at+120]!r} != {wanted['prompt'][at:at+120]!r}")
                if tokenizer.encode(prompt, parse_special=True) != wanted["tokens"]:
                    raise AssertionError("chat token IDs differ")
                counts["chat"] += 1
                for output in ("plain answer", THINK_START + "why" + THINK_END + "answer",
                               THINK_END + "answer"):
                    wanted_msg = call({"op": "parse", "content": output, "partial": False})["message"]
                    parser = M31OutputParser(thinking=kwargs.get("enable_thinking", True), tools=toolset)
                    events = parser.feed(output) + parser.finish()
                    for field, kind in (("content", "content"), ("reasoning_content", "reasoning")):
                        actual = "".join(event.text for event in events if event.kind == kind)
                        if actual != (wanted_msg.get(field) or ""):
                            raise AssertionError(f"parser differs {field}: {actual!r} != {wanted_msg.get(field)!r}")
                    counts["parse"] += 1
                if toolset and {"query", "values"} <= set(toolset[0]["function"].get("parameters", {}).get("properties", {})):
                    output = (THINK_END + "searching" + CALL_START + NS + '<invoke name="lookup">' +
                              NS + "<query>Москва & 東京" + NS + "</query>" + NS + "<values>" +
                              NS + "<item>1" + NS + "</item>" + NS + "<item>2" + NS + "</item>" +
                              NS + "</values>" + NS + "</invoke>" + CALL_END)
                    wanted_msg = call({"op": "parse", "content": output, "partial": False})["message"]
                    parser = M31OutputParser(tools=toolset)
                    events = []
                    for char in output:
                        events.extend(parser.feed(char))
                    events.extend(parser.finish())
                    calls = [event.call for event in events if event.kind == "tool_call"]
                    expected_calls = wanted_msg.get("tool_calls", [])
                    if len(calls) != len(expected_calls):
                        raise AssertionError("tool parser call count differs")
                    for actual, expected in zip(calls, expected_calls):
                        function = expected["function"]
                        if actual.name != function["name"] or actual.arguments != json.loads(function["arguments"]):
                            raise AssertionError("recursive Unicode tool parser differs")
                    counts["parse"] += 1
        return counts
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--oracle", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.model, args.oracle), sort_keys=True))


if __name__ == "__main__":
    main()
