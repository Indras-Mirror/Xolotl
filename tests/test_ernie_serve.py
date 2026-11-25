"""ERNIE-5.5-Flash in the server: serve/ernie.py's prompt (Baidu's renderer) and its output parser.

The round trip is the main check: an assistant turn rendered by Baidu's renderer, fed back to ErnieOutputParser as
the model would write it (after the prompt's <ernie:think>), gives back the same reasoning, answer and calls.
ERNIE_REF_DIR=<the model's HF folder> also checks the vendored renderer against Baidu's encoding_ernie.py byte for byte.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.ernie import (CALL, END, THINK, THINK_END, ErnieOutputParser, ErnieTranslator, effort_of,  # noqa: E402
                         render, to_ernie_force)

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "the weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
FLAT = [t["function"] for t in TOOLS]


def parse(text, thinking=True, tools=None, step=None, reason="stop"):
    p = ErnieOutputParser(thinking=thinking, tools=tools or [], stream_tools=True, recover=True)
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


def assistant_written(message, tools=None):
    """What the model writes for `message`: its rendered assistant body after the prompt's own <ernie:think>."""
    full = render([{"role": "user", "content": "q"}, message, {"role": "user", "content": "next"}], tools)
    head = "<|ernie:im_start|>assistant\n"
    body = full[full.index(head) + len(head):]
    body = body[:body.index(END) + len(END)]
    assert body.startswith(THINK)
    return body[len(THINK):]


class RenderTest(unittest.TestCase):
    def test_generation_prompt(self):
        p = render([{"role": "user", "content": "hi"}])
        self.assertTrue(p.startswith("<|ERNIE_MASK|><|ernie:im_start|>system\n<ernie:global_setting>\nreasoning_effort=high"))
        self.assertTrue(p.endswith("<|ernie:im_start|>user\nhi<|ernie:im_end|>\n<|ernie:im_start|>assistant\n<ernie:think>"))

    def test_thinking_off_and_effort(self):
        off = render([{"role": "user", "content": "hi"}], kwargs={"enable_thinking": False})
        self.assertTrue(off.endswith("assistant\n<ernie:think></ernie:think>"))
        self.assertNotIn("reasoning_effort", off)
        low = render([{"role": "user", "content": "hi"}], kwargs={"reasoning_effort": "low"})
        self.assertIn("reasoning_effort=low", low)
        self.assertIn("reasoning_effort=high", render([{"role": "user", "content": "x"}], kwargs={"reasoning_effort": "max"}))
        self.assertEqual(effort_of({"reasoning_effort": "none"})[0], False)

    def test_tools_and_system(self):
        p = render([{"role": "system", "content": "be brief"}, {"role": "user", "content": "w?"}], TOOLS)
        self.assertIn("<ernie:system_setting>\nbe brief\n</ernie:system_setting>", p)
        self.assertIn("parallel_tool_calls=True", p)
        self.assertIn('<ernie:tool_list>\n[{"function": {"description": "the weather", "name": "get_weather"', p)

    @unittest.skipUnless(os.environ.get("ERNIE_REF_DIR"), "ERNIE_REF_DIR not set")
    def test_vendored_renderer_is_baidus(self):
        ref = Path(os.environ["ERNIE_REF_DIR"]) / "encoding_ernie.py"
        ours = (ROOT / "serve" / "ernie_encoding.py").read_text(encoding="utf-8")
        self.assertTrue(ours.endswith(ref.read_text(encoding="utf-8")))


class ParseTest(unittest.TestCase):
    def test_round_trip_reasoning_answer_calls(self):
        msg = {"role": "assistant", "reasoning_content": "Check the weather.", "content": "Looking it up.",
               "tool_calls": [{"id": "a", "type": "function",
                               "function": {"name": "get_weather", "arguments": {"city": "Melbourne", "days": 3}}}]}
        text = assistant_written(msg, TOOLS)
        for step in (None, 1, 3, 7):
            r, c, calls = parse(text, tools=FLAT, step=step)
            self.assertEqual(r.strip(), "Check the weather.")
            self.assertEqual(c.strip(), "Looking it up.")
            self.assertEqual(calls, [("get_weather", {"city": "Melbourne", "days": 3})], step)

    def test_two_calls_and_special_characters(self):
        args = {"city": 'Ho"bart <&> 中文', "days": 2}
        msg = {"role": "assistant", "content": "",
               "tool_calls": [{"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": args}},
                              {"id": "b", "type": "function",
                               "function": {"name": "get_weather", "arguments": {"city": "Perth"}}}]}
        r, c, calls = parse(assistant_written(msg, TOOLS), tools=FLAT, step=2)
        self.assertEqual(calls, [("get_weather", args), ("get_weather", {"city": "Perth"})])
        self.assertEqual(c.strip(), "")

    def test_thinking_off_reply_is_the_answer(self):
        r, c, calls = parse("Just the answer." + END, thinking=False)
        self.assertEqual((r, c.strip(), calls), ("", "Just the answer.", []))

    def test_a_lone_angle_bracket_is_text(self):
        r, c, _ = parse("a < b" + THINK_END + "x <e y" + END, step=1)
        self.assertEqual(r.strip(), "a < b")
        self.assertEqual(c.strip(), "x <e y")

    def test_cut_call_block_keeps_closed_invokes(self):
        text = "t" + THINK_END + CALL + '\n<invoke name="get_weather"><parameter name="city">Ade</parameter></invoke>\n<inv'
        _, _, calls = parse(text, tools=FLAT, reason="length")
        self.assertEqual(calls, [("get_weather", {"city": "Ade"})])

    def test_translator_clean_points(self):
        t = ErnieTranslator(thinking=True)
        t.feed("abc")
        self.assertTrue(t.clean)
        t.feed("</ernie:th")
        self.assertFalse(t.clean)

    def test_force_spelling(self):
        self.assertEqual(to_ernie_force("<tool_call>\n<function=get_weather>\n"),
                         CALL + '\n<invoke name="get_weather">')


if __name__ == "__main__":
    unittest.main()
