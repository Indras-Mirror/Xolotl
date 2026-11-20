"""tests/test_inkling_serve.py - Inkling-2-Mini in serve/server.py: serve/inkling.py's prompt render and parser.

The render must be the official chat template byte for byte (its jinja2 rendering, generation tags stripped, as
transformers runs it); the parser must turn the model's raw blocks back into OpenAI reasoning_content / content /
tool_calls, and a streamed parse must end the same as a one-shot one.

    python -m pytest -q tests/test_inkling_serve.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.inkling import (HEADER, InklingOutputParser, effort_value, render,  # noqa: E402
                           to_inkling_force)
from serve.server import Service  # noqa: E402

TEMPLATE = Path(os.environ.get(
    "INKLING_TEMPLATE", "/media/mal/NVME1TB/Xolotl/inkling-2-mini/Inkling-2-Mini/chat_template.jinja"))

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search", "description": "Search the web.",
    "parameters": {"type": "object", "properties": {
        "q": {"type": "string"}, "limit": {"type": "integer"}}}}}
TOOLS = [WEATHER, SEARCH]
SCHEMAS = [t["function"] for t in TOOLS]

CONV_A = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Melbourne for 3 days?"},
    {"role": "assistant", "reasoning_content": "The user asks for the weather.", "content": "Let me check.",
     "tool_calls": [{"id": "call_1", "type": "function", "function": {
         "name": "get_weather", "arguments": {"city": "Melbourne", "days": 3}}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 24C"},
    {"role": "assistant", "reasoning_content": "Now I know.", "content": "It is sunny and 24C in Melbourne."},
]

CONV_B = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": [{"type": "text", "text": "Name three primes."}]},
    {"role": "assistant", "reasoning_content": "2, 3, 5.", "content": "2, 3, 5."},
]

CONV_C = [
    {"role": "system", "content": "You are a research agent."},
    {"role": "user", "content": "Weather in Hobart and search for penguins."},
    {"role": "assistant", "content": "",
     "tool_calls": [
         {"id": "a", "type": "function", "function": {
             "name": "get_weather", "arguments": {"city": "Hobart", "zip": None,
                                                  "tags": ["cold", "wet"], "meta": {"ok": True, "n": 2.5}}}},
         {"id": "b", "type": "function", "function": {
             "name": "search", "arguments": {"q": "penguins \"Tasmania\" & 中文", "limit": 5}}}]},
    {"role": "tool", "name": "get_weather", "content": "Cold, 9C"},
    {"role": "tool", "name": "search", "content": "Little penguins."},
    {"role": "user", "content": "Thanks. Summarise in one line."},
    {"role": "assistant", "content": "Hobart is cold; penguins live there."},
]


def official_render(messages, tools=None, reasoning_effort=None):
    """The model's chat_template.jinja as transformers renders it: HF's generation tags are stripped (they only mark
    the assistant spans for the loss mask) and its tojson is the chat template's."""
    import jinja2

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise ValueError(message)

    src = re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "", TEMPLATE.read_text(encoding="utf-8"))
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(src).render(messages=messages, tools=tools, reasoning_effort=reasoning_effort)


def assistant_raw(message, tools=None):
    """The blocks the model writes for `message` (the assistant part of its rendered prompt)."""
    full = render([{"role": "system", "content": "s"}, message], tools)
    return full[full.index(HEADER):]


def parse(text, thinking=True, tools=None, step=None, reason="stop"):
    p = InklingOutputParser(thinking=thinking, tools=tools or [], stream_tools=True, recover=True)
    evs = []
    if step:
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
    else:
        evs += p.feed(text)
    evs += p.finish(reason)
    reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
    content = "".join(e.text for e in evs if e.kind == "content")
    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
    return reasoning, content, calls


@unittest.skipUnless(TEMPLATE.exists(), f"the official chat template is not at {TEMPLATE}")
class RenderParityTest(unittest.TestCase):
    def test_conversation_with_a_tool_call_and_result(self):
        for effort in (None, "low", 0.0):
            self.assertEqual(render(CONV_A, TOOLS, effort), official_render(CONV_A, TOOLS, effort))

    def test_content_parts_and_effort_names(self):
        for effort in ("low", "medium", "xhigh", "none", 0.55):
            self.assertEqual(render(CONV_B, None, effort), official_render(CONV_B, None, effort))

    def test_two_calls_in_one_turn(self):
        for effort in (None, 0.0):
            self.assertEqual(render(CONV_C, TOOLS, effort), official_render(CONV_C, TOOLS, effort))

    def test_anthropic_style_tools(self):
        tools = [{"name": "search", "description": "Search.", "parameters": {"type": "object", "properties": {}}}]
        messages = [{"role": "user", "content": "find it"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c", "type": "function", "function": {"name": "search", "arguments": {"q": "x"}}}]}]
        self.assertEqual(render(messages, tools), official_render(messages, tools))

    def test_effort_block_is_the_templates(self):
        prompt = render(CONV_B, None, "none")
        self.assertIn("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>", prompt)
        self.assertIn("<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>",
                      render(CONV_B, None, "low"))

    def test_string_arguments_are_refused_like_the_template(self):
        message = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]}
        with self.assertRaises(ValueError):
            render([{"role": "user", "content": "u"}, message])
        with self.assertRaises(ValueError):
            official_render([{"role": "user", "content": "u"}, message])


class ParserTest(unittest.TestCase):
    def test_thinking_only(self):
        message = {"role": "assistant", "reasoning_content": "Just thinking.", "content": ""}
        self.assertEqual(parse(assistant_raw(message)), ("Just thinking.", "", []))

    def test_text_only(self):
        message = {"role": "assistant", "content": "Hello there."}
        self.assertEqual(parse(assistant_raw(message)), ("", "Hello there.", []))

    def test_text_only_without_thinking(self):
        self.assertEqual(parse(assistant_raw({"role": "assistant", "content": "Hello there."}), thinking=False),
                         ("", "Hello there.", []))

    def test_one_tool_call(self):
        message = {"role": "assistant", "reasoning_content": "Need it.", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Melbourne", "days": 3}}}]}
        self.assertEqual(parse(assistant_raw(message), tools=SCHEMAS),
                         ("Need it.", "", [("get_weather", {"city": "Melbourne", "days": 3})]))

    def test_two_tool_calls(self):
        message = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Hobart", "days": 2}}},
            {"id": "b", "type": "function", "function": {
                "name": "search", "arguments": {"q": "penguins", "limit": 5}}}]}
        self.assertEqual(parse(assistant_raw(message), tools=SCHEMAS), ("", "", [
            ("get_weather", {"city": "Hobart", "days": 2}), ("search", {"q": "penguins", "limit": 5})]))

    def test_arguments_with_special_characters(self):
        args = {"text": "line1\nline2 \"quoted\" <tag> & 中文 = 1+1",
                "nested": {"a": [1, 2], "ok": True, "none": None}, "n": 2.5}
        message = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "search", "arguments": args}}]}
        schema = [{"name": "search", "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "nested": {"type": "object"}, "n": {"type": "number"}}}}]
        self.assertEqual(parse(assistant_raw(message), tools=schema), ("", "", [("search", args)]))

    def test_streaming_matches_one_shot(self):
        messages = [
            {"role": "assistant", "reasoning_content": "Think.", "content": "Answer."},
            {"role": "assistant", "reasoning_content": "Need it.", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "get_weather", "arguments": {"city": "Hobart", "days": 3}}}]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "search", "arguments": {"q": "x&y <z>"}}},
                {"id": "b", "type": "function", "function": {
                    "name": "get_weather", "arguments": {"city": "Perth"}}}]},
        ]
        for message in messages:
            raw = assistant_raw(message)
            whole = parse(raw, tools=SCHEMAS)
            for step in (1, 2, 3, 7, 13):
                self.assertEqual(parse(raw, tools=SCHEMAS, step=step), whole, (step, raw))

    def test_cut_inside_a_marker_drops_it(self):
        raw = HEADER + "<|content_text|>Hi<|end_message|><|content_model_end_sa"
        self.assertEqual(parse(raw, thinking=False, reason="length"), ("", "Hi", []))

    def test_answer_after_the_thinking_block_is_content(self):
        raw = (HEADER + "<|content_thinking|>Hmm.<|end_message|>"
               + HEADER + "<|content_text|>The answer.<|end_message|>")
        self.assertEqual(parse(raw), ("Hmm.", "The answer.", []))

    def test_undeclared_call_is_still_a_call(self):
        raw = HEADER + 'f<|content_invoke_tool_json|>{"name":"f","args":{"x":1}}<|end_message|>'
        self.assertEqual(parse(raw, tools=[]), ("", "", [("f", {"x": 1})]))

    def test_clean_points(self):
        """The server writes a forced call or a thinking close only where the template would: after a block's end."""
        from serve.inkling import InklingTranslator
        t = InklingTranslator(thinking=True)
        t.feed(HEADER)
        self.assertFalse(t.clean)                          # a name may be arriving
        t.feed("<|content_thinking|>thinking")
        self.assertTrue(t.clean)                           # the thinking budget may close it here
        t.feed("<|end_message|>")
        self.assertTrue(t.clean)                           # a block boundary
        t.feed(HEADER + "<|content_text|>answer")
        self.assertFalse(t.clean)                          # the answer block is open
        t.feed("<|end_message|>")
        self.assertTrue(t.clean)
        t.feed(HEADER + 'get_weather<|content_invoke_tool_json|>{"name":')
        self.assertFalse(t.clean)                          # inside a call's JSON


class ForceTest(unittest.TestCase):
    def test_named_tool(self):
        self.assertEqual(to_inkling_force("<tool_call>\n<function=get_weather>\n"),
                         '<|message_model|>get_weather<|content_invoke_tool_json|>'
                         '{"name":"get_weather","args":{')

    def test_any_tool(self):
        self.assertEqual(to_inkling_force("<tool_call>\n<function="), "<|message_model|>")
        self.assertEqual(to_inkling_force(None), None)

    def test_effort_values(self):
        self.assertEqual(effort_value(None), 0.9)
        self.assertEqual(effort_value("none"), 0.0)
        self.assertEqual(effort_value("xhigh"), 0.99)
        self.assertEqual(effort_value(0.5), 0.5)
        for bad in ("warp", 1.0, -0.1):
            with self.assertRaises(ValueError):
                effort_value(bad)


class FamilyHookTest(unittest.TestCase):
    def service(self, family):
        svc = Service.__new__(Service)
        svc.tok = type("Tok", (), {"special_ids": {"eos": 200006}})()
        svc.set_family(family)
        return svc

    def test_family_sets_the_parser_and_end_token(self):
        from serve.inkling import InklingOutputParser
        for family in ("inkling2", "inkling2_loop"):
            svc = self.service(family)
            self.assertIs(svc.parser_cls, InklingOutputParser)
            self.assertEqual(svc.stop_ids, {200006})
            self.assertEqual(svc.answer_open_text, HEADER + "<|content_text|>")
            self.assertIsNotNone(svc.reasoning_wrap_up_text)

    def test_effort_kwargs_reach_the_template(self):
        svc = self.service("inkling2_loop")
        prompt = svc.render_prompt([{"role": "user", "content": "hi"}], None, {"reasoning_effort": "low"})
        self.assertIn("Thinking effort level: 0.2", prompt)
        prompt = svc.render_prompt([{"role": "user", "content": "hi"}], None, {"enable_thinking": False})
        self.assertIn("Thinking effort level: 0", prompt)


if __name__ == "__main__":
    unittest.main()
