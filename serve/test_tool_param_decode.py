"""Tool-call parameters are decoded by their declared schema: text stays text where a string is allowed."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import OutputParser, parse_tool_call  # noqa: E402

NULLABLE = {"anyOf": [{"type": "string", "minLength": 1, "maxLength": 1000}, {"type": "null"}]}
TOOL = {"name": "write_note", "parameters": {"type": "object", "properties": {
    "path": NULLABLE,
    "one": {"oneOf": [{"type": "string"}, {"type": "null"}]},
    "only": {"anyOf": [{"type": "string"}]},
    "listed": {"type": ["string", "null"]},
    "plain": {"type": "string"},
    "mixed": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
    "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    "n": {"type": "integer"},
    "free": {},
}}}


def call(name, value):
    return f"<function=write_note>\n<parameter={name}>\n{value}\n</parameter>\n</function>"


class ParseToolCall(unittest.TestCase):
    def arg(self, name, value):
        return parse_tool_call(call(name, value), TOOL).arguments[name]

    def test_nullable_string_stays_text(self):
        for name in ("path", "one", "only", "listed", "plain"):
            self.assertEqual(self.arg(name, "notes/x.txt"), "notes/x.txt", name)
            self.assertEqual(self.arg(name, "123"), "123", name)
            self.assertEqual(self.arg(name, "true"), "true", name)
            self.assertEqual(self.arg(name, '["a"]'), '["a"]', name)

    def test_null_is_null_only_where_null_is_a_type(self):
        for name in ("path", "one", "listed"):
            self.assertIsNone(self.arg(name, "null"), name)
        self.assertEqual(self.arg("only", "null"), "null")
        self.assertEqual(self.arg("plain", "null"), "null")

    def test_multiline_text(self):
        self.assertEqual(self.arg("path", "a\nb"), "a\nb")

    def test_non_text_types_are_json(self):
        self.assertEqual(self.arg("count", "7"), 7)
        self.assertIsNone(self.arg("count", "null"))
        self.assertEqual(self.arg("n", "5"), 5)
        self.assertEqual(self.arg("free", "[1, 2]"), [1, 2])

    def test_two_parameters(self):
        body = ("<function=write_note>\n<parameter=path>\nnotes/x.txt\n</parameter>\n"
                "<parameter=one>\n42\n</parameter>\n</function>")
        self.assertEqual(parse_tool_call(body, TOOL).arguments, {"path": "notes/x.txt", "one": "42"})


class StreamedCall(unittest.TestCase):
    def run_parser(self, text, step):
        p = OutputParser(thinking=False, tools=[TOOL], stream_tools=True)
        pieces, calls = [], []
        events = []
        for i in range(0, len(text), step):
            events += p.feed(text[i:i + step])
        events += p.finish()
        for e in events:
            if e.kind == "tool_args":
                pieces.append(e.text)
            elif e.kind == "tool_call":
                calls.append(e.call)
        return "".join(pieces), calls

    def check(self, name, value, want):
        text = "<tool_call>\n" + call(name, value) + "\n</tool_call>"
        for step in (1, 7, len(text)):
            streamed, calls = self.run_parser(text, step)
            self.assertEqual(json.loads(streamed), {name: want}, (name, value, step))
            self.assertEqual(calls[0].arguments, {name: want}, (name, value, step))

    def test_streamed_matches_final(self):
        self.check("path", "notes/x.txt", "notes/x.txt")
        self.check("path", "123", "123")
        self.check("path", "null", None)
        self.check("plain", "123", "123")
        self.check("n", "5", 5)
        self.check("count", "null", None)


UNION = {"name": "automation", "parameters": {"type": "object", "properties": {},
    "oneOf": [{"$ref": "#/$defs/view"}, {"$ref": "#/$defs/write"}],
    "$defs": {
        "id": {"type": "string"},
        "view": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["view"]},
                                                 "id": {"$ref": "#/$defs/id"}}},
        "write": {"oneOf": [{"$ref": "#/$defs/create"}]},
        "create": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["create"]},
                                                   "name": {"type": "string"}, "every": {"type": "integer"}}},
    }}}


class UnionParameters(unittest.TestCase):
    """Parameters sent as a oneOf union are typed by the variant that declares them."""

    def test_each_variant_types_its_parameters(self):
        body = ("<function=automation>\n<parameter=mode>\ncreate\n</parameter>\n"
                "<parameter=name>\n123\n</parameter>\n<parameter=every>\n8\n</parameter>\n</function>")
        self.assertEqual(parse_tool_call(body, UNION).arguments, {"mode": "create", "name": "123", "every": 8})
        body = ("<function=automation>\n<parameter=mode>\nview\n</parameter>\n"
                "<parameter=id>\n42\n</parameter>\n</function>")
        self.assertEqual(parse_tool_call(body, UNION).arguments, {"mode": "view", "id": "42"})

    def test_streamed(self):
        text = ("<tool_call>\n<function=automation>\n<parameter=name>\n123\n</parameter>\n"
                "<parameter=every>\n8\n</parameter>\n</function>\n</tool_call>")
        for step in (1, 7, len(text)):
            p = OutputParser(thinking=False, tools=[UNION], stream_tools=True)
            events = []
            for i in range(0, len(text), step):
                events += p.feed(text[i:i + step])
            events += p.finish()
            streamed = "".join(e.text for e in events if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), {"name": "123", "every": 8}, step)


if __name__ == "__main__":
    unittest.main()
