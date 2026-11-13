"""serve/test_atem.py - Muse Glimmer's turn format (serve/atem.py): channels and ATEM calls -> the parser's events.

    python -m unittest serve.test_atem -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.atem import AtemOutputParser, AtemTranslator, to_atem_force  # noqa: E402

WEATHER = [{"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}}}}]

THINK_ANSWER = " to=self<|message|>Let me think.<|eom|><|start|>assistant to=user<|message|>The answer is 4."
THINK_CALL = (' to=self<|message|>Need weather.<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n'
              '<atem:invoke name="get_weather">\n<atem:parameter name="city">Melbourne</atem:parameter>\n'
              '<atem:parameter name="days">3</atem:parameter>\n</atem:invoke>\n</atem:function_calls>')
ANSWER_THEN_CALL = ('Done <tag> ok<|eom|><|start|>assistant to=f<|message|><atem:function_calls>\n<atem:invoke name="f">\n'
                    '<atem:parameter name="code">if a<b:\n  print("</x>")</atem:parameter>\n</atem:invoke>\n</atem:function_calls>')


def translate(text, thinking=True, step=1):
    t = AtemTranslator(thinking)
    return "".join(t.feed(text[i:i + step]) for i in range(0, len(text), step)) + t.flush()


def parse(text, thinking=True, tools=None, reason="stop", step=1):
    p = AtemOutputParser(thinking=thinking, tools=tools or [], stream_tools=True, recover=True)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish(reason)
    reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
    content = "".join(e.text for e in evs if e.kind == "content")
    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
    return reasoning, content, calls


class TranslatorTest(unittest.TestCase):
    def test_streaming_split_does_not_matter(self):
        for text, thinking in ((THINK_ANSWER, True), (THINK_CALL, True), (ANSWER_THEN_CALL, False)):
            whole = translate(text, thinking, step=len(text))
            for step in (1, 2, 3, 7, 13):
                self.assertEqual(translate(text, thinking, step), whole)

    def test_think_then_answer(self):
        self.assertEqual(translate(THINK_ANSWER), "Let me think.</think>\n\nThe answer is 4.")

    def test_no_recipient_is_the_answer(self):
        self.assertEqual(translate("<|message|>plain"), "</think>\n\nplain")

    def test_lt_in_answer_is_text(self):
        self.assertEqual(translate(" to=user<|message|>x<y and a<b>c"), "</think>\n\nx<y and a<b>c")


class ParserTest(unittest.TestCase):
    def test_reasoning_and_answer(self):
        self.assertEqual(parse(THINK_ANSWER), ("Let me think.", "The answer is 4.", []))

    def test_call_with_typed_arguments(self):
        r, c, calls = parse(THINK_CALL, tools=WEATHER)
        self.assertEqual((r, c.strip()), ("Need weather.", ""))
        self.assertEqual(calls, [("get_weather", {"city": "Melbourne", "days": 3})])

    def test_answer_then_call_value_keeps_markup(self):
        r, c, calls = parse(ANSWER_THEN_CALL, thinking=False, tools=[{"name": "g", "parameters": {}}])
        self.assertEqual(c.strip(), "Done <tag> ok")
        self.assertEqual(calls, [("f", {"code": 'if a<b:\n  print("</x>")'})])

    def test_thinking_only_becomes_the_answer(self):
        self.assertEqual(parse(" to=self<|message|>Just this.")[1], "Just this.")

    def test_cut_inside_a_tag_drops_it(self):
        _, c, calls = parse(' to=user<|message|>Hi<|eom|><|start|>assistant to=f<|message|><atem:invoke na',
                            reason="length")
        self.assertEqual((c, calls), ("Hi", []))

    def test_call_without_message_marker(self):
        text = (' to=self<|message|>ok<|eom|><|start|>assistant to=get_weather\n\n<atem:function_calls>\n'
                '<atem:invoke name="get_weather">\n<atem:parameter name="city">Hobart</atem:parameter>\n</atem:invoke>\n'
                '</atem:function_calls>')
        for step in (1, 5):
            self.assertEqual(parse(text, tools=WEATHER, step=step)[2], [("get_weather", {"city": "Hobart"})])

    def test_force(self):
        self.assertEqual(to_atem_force("<tool_call>\n<function=get_weather>\n"),
                         ' to=get_weather<|message|><atem:function_calls>\n<atem:invoke name="get_weather">\n')


if __name__ == "__main__":
    unittest.main()
