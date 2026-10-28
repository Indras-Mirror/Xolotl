#!/usr/bin/env python3
"""Check the common HTTP adapter's actual IDs against native vocabulary-only grammar.

This optional check uses the model header, the exported GGUF template and the
same pinned native grammar as m31_generate. It never runs a model decoder.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools/m31/serve")]
from gguf_reader import GGUFFile  # noqa: E402
from m31_tok import M31Tokenizer  # noqa: E402
import m31_parser as P  # noqa: E402
import serve_m31 as M  # noqa: E402
from serve import server as S  # noqa: E402


def make_cases(model, directory):
    metadata = GGUFFile(Path(model)).metadata
    tok = M31Tokenizer(metadata["tokenizer.ggml.tokens"], metadata["tokenizer.ggml.merges"],
                       metadata["tokenizer.ggml.token_type"])
    template_path = directory / "chat_template.jinja"
    template_path.write_text(metadata["tokenizer.chat_template"], encoding="utf-8")
    service = M.M31Service(S.MockEngine(tok, "", max_context=65536), tok, M.M31Template(template_path))
    properties = {"query": {"type": "string"}, "options": {"type": "object", "properties": {
        "cities": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "count": {"type": "integer"}}, "required": ["name", "count"]}},
        "enabled": {"type": "boolean"}}, "required": ["cities", "enabled"]}}
    tools = [{"name": "lookup", "parameters": {"type": "object", "properties": properties, "required": ["query", "options"]}},
             {"name": "noop", "parameters": {"type": "object", "properties": {}}}]
    tag = lambda name, value: P.NS + "<" + name + ">" + value + P.NS + "</" + name + ">"
    output = (P.CALL_START + "\n" + P.NS + '<invoke name="lookup">' + tag("query", "Москва & 東京") +
              tag("options", tag("cities", tag("item", tag("name", "München") + tag("count", "2"))) +
                  tag("enabled", "true")) + P.INVOKE_END + P.CALL_END)
    cases = []
    messages = [{"role": "user", "content": "Find these cities."}]
    for choice in ("auto", "required", {"type": "function", "function": {"name": "lookup"}}, "none"):
        request = {"tool_choice": choice}
        offered = tools if choice != "none" else []
        force = P.forced_call(choice, offered)
        ids, _, _ = service.prepare(messages, offered, {"enable_thinking": False}, 256, force=force, req=request)
        suffix = output[len(force):] if force else output
        if choice == "none":
            suffix = "plain answer"
        name = choice if isinstance(choice, str) else "named"
        cases.append({"name": "tool-" + name, "chat_b64": M.chat_key(ids.chat_request), "ids": list(ids),
                      "active": choice != "none", "accept_ids": tok.encode(suffix, parse_special=True),
                      "accept_eog_after": True, "reject_eog_before": name in ("required", "named")})
        if choice != "none":
            generated = tok.encode(suffix, parse_special=True)
            cut = max(1, len(generated) // 2)
            continuation = copy.deepcopy(cases[-1])
            continuation.update(name="continued-" + name, ids=list(ids) + generated[:cut],
                                accept_ids=generated[cut:], reject_eog_before=False)
            cases.append(continuation)
    schema = {"type": "object", "properties": {"city": {"type": "string", "enum": ["東京", "München"]}},
              "required": ["city"], "additionalProperties": False}
    for format_ in ({"type": "json_object"}, {"type": "json_schema", "json_schema": {"name": "City", "schema": schema}}):
        prepared_messages, _ = S.prepare_format(format_, messages)
        ids, _, _ = service.prepare(prepared_messages, [], {"enable_thinking": False}, 256,
                                    req={"response_format": format_})
        cases.append({"name": format_["type"], "chat_b64": M.chat_key(ids.chat_request), "ids": list(ids),
                      "active": True, "accept_ids": tok.encode('```json\n{"city":"東京"}\n```', parse_special=True),
                      "reject_eog_before": True, "accept_eog_after": True})
    raw = copy.deepcopy(ids.chat_request)
    raw["grammar"] = 'root ::= "x"'
    cases.append({"name": "raw-gbnf-rejected", "chat_b64": M.chat_key(raw), "ids": list(ids), "expect_error": True})
    bad = copy.deepcopy(cases[0])
    bad.update(name="mismatched-prompt-rejected", ids=[1], expect_error=True)
    cases.append(bad)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new evidence directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = make_cases(args.model, args.out)
    path = args.out / "cases.jsonl"
    path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with (args.out / "native.stderr.log").open("w") as error:
        result = subprocess.run([str(args.check.resolve()), str(args.model.resolve()), str(path.resolve())],
                                capture_output=False, stdout=subprocess.PIPE, stderr=error, text=True,
                                env=env, timeout=300)
    (args.out / "native.jsonl").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise SystemExit(f"native grammar check failed ({result.returncode}); see {args.out}")
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    if len(rows) != len(cases) or not all(row.get("ok") is True for row in rows):
        raise SystemExit("native grammar check did not report every case")
    print(json.dumps({"cases": len(cases), "vocab_only": True, "decoder": False, "ok": True}))


if __name__ == "__main__":
    main()
