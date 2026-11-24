"""serve/test_restore_reasoning.py - an answer sent back without its thinking gets the thinking this server wrote for it
(mock engine, no GPU).

The template keeps a turn's thinking (preserve_thinking), and OpenAI chat clients drop reasoning_content from the
history, so the next prompt left the conversation cache at that turn and the whole answer was read again.

    python -m unittest serve.test_restore_reasoning -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class RestoreReasoning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, "<think>\nmy private plan\n</think>\n\nthe answer", max_context=8192)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.recent_reasoning.clear()

    def chat(self, messages, stream=False):
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps({"messages": messages, "stream": stream}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return self.tok.decode(self.engine.last_prompt)

    def test_dropped_thinking_comes_back(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.svc.recent_reasoning.clear()
                self.chat([{"role": "user", "content": "q1"}], stream)
                prompt = self.chat([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "the answer"},
                                    {"role": "user", "content": "q2"}], stream)
                self.assertIn("<think>\nmy private plan\n</think>\n\nthe answer", prompt)

    def test_client_thinking_and_unknown_answers_stay(self):
        self.chat([{"role": "user", "content": "q1"}])
        prompt = self.chat([{"role": "user", "content": "q1"},
                            {"role": "assistant", "content": "the answer", "reasoning_content": "theirs"},
                            {"role": "user", "content": "q2"}])
        self.assertIn("<think>\ntheirs\n</think>\n\nthe answer", prompt)
        prompt = self.chat([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "another answer"},
                            {"role": "user", "content": "q2"}])
        self.assertIn("<think>\n\n</think>\n\nanother answer", prompt)


class RestoreReasoningSameAnswer(unittest.TestCase):
    """Two replies with the same answer and different thinking: each turn of the history gets its own back (one key
    for both gave the older turn the newer thinking, and the prompt left the cache at the older turn)."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, ["<think>\nplan one\n</think>\n\nok", "<think>\nplan two\n</think>\n\nok"],
                                max_context=8192)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.recent_reasoning.clear()
        self.engine.turns, self.engine.script = 0, self.engine.scripts[0]

    def chat(self, messages):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps({"messages": messages}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return self.tok.decode(self.engine.last_prompt)

    def test_each_turn_gets_its_own_thinking(self):
        u = lambda s: {"role": "user", "content": s}
        ok = {"role": "assistant", "content": "ok"}
        self.chat([u("first question")])
        self.chat([u("first question"), ok, u("second question")])
        prompt = self.chat([u("first question"), ok, u("second question"), ok, u("third question")])
        one, two = prompt.index("plan one"), prompt.index("plan two")
        self.assertLess(prompt.index("first question"), one)
        self.assertLess(one, prompt.index("second question"))
        self.assertLess(prompt.index("second question"), two)

    def test_a_turn_without_its_own_thinking_gets_none(self):
        u = lambda s: {"role": "user", "content": s}
        ok = {"role": "assistant", "content": "ok"}
        self.chat([u("question x")])   # this reply's thinking is remembered with "question x" before it
        prompt = self.chat([u("question y"), ok, u("question z")])   # an "ok" that no reply here wrote
        self.assertNotIn("plan", prompt)


class RestoreReasoningToolCall(unittest.TestCase):
    """A tool-call turn has no answer text: its thinking is found by the call (name and arguments)."""
    TOOLS = [{"type": "function", "function": {"name": "get", "description": "d",
                                               "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, "<think>\nplan the call\n</think>\n\n<tool_call>\n<function=get>\n<parameter=q>\nx\n"
                                         "</parameter>\n</function>\n</tool_call>", max_context=8192)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def chat(self, messages):
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps({"messages": messages, "tools": self.TOOLS}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
        return body, self.tok.decode(self.engine.last_prompt)

    def test_tool_call_turn_gets_its_thinking(self):
        body, _ = self.chat([{"role": "user", "content": "find x"}])
        calls = body["choices"][0]["message"]["tool_calls"]
        _, prompt = self.chat([{"role": "user", "content": "find x"},
                               {"role": "assistant", "content": None, "tool_calls": calls},
                               {"role": "tool", "tool_call_id": calls[0]["id"], "content": "x is 1"}])
        self.assertIn("<think>\nplan the call\n</think>", prompt)


class RestoreReasoningRepeatedCall(unittest.TestCase):
    """An agent loop repeats one tool call after the same tool output: each repeat keeps the thinking it was written
    with from one request to the next (the newest-first lookup gave every older repeat the newest thinking, so each new
    repeat rewrote a turn in the middle of the history and the engine read the whole prompt again)."""

    def test_repeats_keep_their_thinking(self):
        svc = Service.__new__(Service)
        svc.recent_reasoning, svc.reasoning_mu = __import__("collections").OrderedDict(), __import__("threading").Lock()
        call = [{"id": "c", "type": "function", "function": {"name": "Bash", "arguments": json.dumps({"command": "adb devices"})}}]
        key = Service._answer_key("", call)
        out = "List of devices attached\nemulator-5554\tdevice\n"
        tools = ["starting\n" + out, out, out, out, out]

        def conv(n):
            m = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
            for i in range(n):
                m += [{"role": "assistant", "content": "", "tool_calls": call}, {"role": "tool", "content": tools[i]}]
            return m

        seen = []
        for i in range(len(tools)):
            before = "go" if i == 0 else tools[i - 1]
            svc.recent_reasoning.setdefault(key, []).append(("<|tool|>" + before + "<|end|><|model|>", f"think {i}"))
            m = conv(i + 1)
            svc._restore_reasoning_locked(m, False)
            got = [x.get("reasoning_content") for x in m if x["role"] == "assistant"]
            self.assertEqual(got, [f"think {k}" for k in range(i + 1)])   # every turn its own, the newest included
            for old in seen:                                               # and no earlier turn ever changes
                self.assertEqual(got[:len(old)], old)
            seen.append(got)


if __name__ == "__main__":
    unittest.main()
