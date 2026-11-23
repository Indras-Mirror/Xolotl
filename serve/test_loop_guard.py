"""serve/test_loop_guard.py - the tool-call loop guard (serve/loop_guard.py): what counts as the same call
N times in a row, the note it adds before the model's turn, and that every other request renders as before.

    python -m unittest serve.test_loop_guard -v
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import unittest
from unittest import mock
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import loop_guard as LG  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "v41"))
import dsv41_encoding as E  # noqa: E402  (DeepSeek V4.1's own encoder: pure Python)

TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}}}}]


def call(name="bash", args=None, cid="c"):
    args = {"command": "pytest -x"} if args is None else args
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str)
                                                         else json.dumps(args)}}


WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi".split()


def changing(k, j):
    """A tool result that is new each time (not only by its numbers): no loop over the window."""
    return f"failed in test_{WORDS[k]}_{WORDS[j]}"


def agent(*turns, user="fix the failing test", tail=None, result="1 failed"):
    """An OpenAI chat request: the user's message, then for each turn an assistant message with its calls and one tool
    result per call.  A turn is a list of calls, or a string (an assistant message without a call).  `result`: the
    text of every result, or a function (turn, call) -> its text."""
    msgs = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": user}]
    for k, turn in enumerate(turns):
        if isinstance(turn, str):
            msgs.append({"role": "assistant", "content": turn})
            continue
        calls = [dict(c, id=f"call_{k}_{j}") for j, c in enumerate(turn)]
        msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        msgs += [{"role": "tool", "tool_call_id": c["id"], "content": result(k, j) if callable(result) else result}
                 for j, c in enumerate(calls)]
    if tail:
        msgs += tail
    return {"model": "x", "max_tokens": 50, "tools": TOOLS, "messages": msgs}


def template_messages(req):
    return openai_to_messages(req)[0]


class Repeated(unittest.TestCase):
    def test_four_in_a_row(self):
        self.assertEqual(LG.repeated(template_messages(agent(*[[call()]] * 4))), ("bash", 4))
        self.assertEqual(LG.repeated(template_messages(agent(*[[call()]] * 6))), ("bash", 6))

    def test_three_is_not_a_loop(self):
        self.assertIsNone(LG.repeated(template_messages(agent(*[[call()]] * 3))))

    def test_configured_repeats(self):
        msgs = template_messages(agent(*[[call()]] * 3))
        self.assertEqual(LG.repeated(msgs, 3), ("bash", 3))
        self.assertIsNone(LG.repeated(msgs, 0))                           # off

    def test_other_arguments_break_the_run(self):
        turns = [[call()], [call(args={"command": "pytest -x -k one"})], [call()], [call()], [call()]]
        self.assertIsNone(LG.repeated(template_messages(agent(*turns))))
        self.assertEqual(LG.repeated(template_messages(agent(*turns, [call()]))), ("bash", 4))

    def test_other_tool_breaks_the_run(self):
        turns = [[call()]] * 2 + [[call("read", {"command": "pytest -x"})]] + [[call()]] * 3
        self.assertIsNone(LG.repeated(template_messages(agent(*turns))))

    def test_a_reply_without_a_call_breaks_the_run(self):
        turns = [[call()]] * 2 + ["Still failing; let me look again."] + [[call()]] * 2
        self.assertIsNone(LG.repeated(template_messages(agent(*turns))))

    def test_argument_spelling_does_not_matter(self):
        """A client's JSON string with other spacing or key order is the same call (Maya's args_key)."""
        a = call(args={"command": "ls", "timeout": 5})
        b = call(args='{"timeout": 5,   "command": "ls"}')
        msgs = template_messages(agent([a], [b], [a], [b]))
        self.assertEqual(LG.repeated(msgs), ("bash", 4))
        self.assertEqual(LG.args_key('{"a": 1}'), LG.args_key({"a": 1}))
        self.assertEqual(LG.args_key("not json"), LG.args_key("not json"))

    def test_a_batch_repeated(self):
        batch = [call(), call("read", {"path": "a.py"})]
        self.assertEqual(LG.repeated(template_messages(agent(*[batch] * 4))), ("bash, read", 4))
        self.assertIsNone(LG.repeated(template_messages(agent(*[batch] * 3, [call()]))))

    def test_only_when_the_request_answers_the_call(self):
        """The user wrote after the loop: no note (they are steering); the results with a user text beside them (Claude
        Code's reminders): still the loop."""
        stop = [{"role": "user", "content": "stop repeating yourself"}]
        req = agent(*[[call()]] * 4)
        req["messages"] = req["messages"][:-1]                            # the last call's result not sent yet
        self.assertIsNone(LG.repeated(template_messages(req)))
        msgs = template_messages(agent(*[[call()]] * 4))
        self.assertIsNone(LG.repeated(msgs + [{"role": "assistant", "content": "Done."}] + stop))
        self.assertEqual(LG.repeated(msgs + [{"role": "user", "content": "<system-reminder>x</system-reminder>"}]),
                         ("bash", 4))
        self.assertIsNone(LG.repeated([]))
        self.assertIsNone(LG.repeated(template_messages({"messages": [{"role": "user", "content": "hi"}]})))

    def test_anthropic_conversation(self):
        msgs = [{"role": "user", "content": "fix it"}]
        for k in range(4):
            msgs.append({"role": "assistant", "content": [
                {"type": "text", "text": "Running the tests."},
                {"type": "tool_use", "id": f"t{k}", "name": "Bash", "input": {"command": "npm test"}}]})
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{k}",
                                                      "content": "1 failing"}]})
        got = LG.repeated(anthropic_to_messages({"messages": msgs, "max_tokens": 10})[0])
        self.assertEqual(got, ("Bash", 4))

    def test_setting(self):
        self.assertEqual(LG.setting(None), 4)
        self.assertEqual(LG.setting(True), 4)
        self.assertEqual(LG.setting(False), 0)
        self.assertEqual(LG.setting({"repeats": 3}), 3)
        self.assertEqual(LG.setting({}), 4)
        for bad in ({"repeats": 1}, {"repeats": True}, {"repeats": "4"}, {"times": 4}, "on", 4):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                LG.setting(bad)


class QwenPrompt(unittest.TestCase):
    """The server's own (Qwen) template refuses a late system message: the note is a user turn there."""

    def svc(self, repeats=LG.REPEATS):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=1 << 20), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.loop_guard_repeats = repeats
        return svc

    def prompt(self, svc, req):
        messages, tools, kw = openai_to_messages(req)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ids, _, _ = svc.prepare(messages, tools, kw, 10)
        return svc.tok.decode(ids), out.getvalue()

    def test_no_repeats_render_as_before(self):
        """Byte for byte: guard on and guard off give the same ids for every request that is not a loop (results that
        change each time: no loop over the window either)."""
        reqs = [agent(*[[call()]] * 3, result=changing),
                agent([call()], [call("read", {"path": "x"})], [call()], [call()], result=changing),
                agent(*[[call()]] * 2 + ["thinking out loud"] + [[call()]] * 3, result=changing),
                agent([call()], [call("read", {"path": "x"})], [call()], [call("read", {"path": "y"})]),
                {"model": "x", "messages": [{"role": "user", "content": "hi"}]}]
        for req in reqs:
            with self.subTest(n=len(req["messages"])):
                on, log = self.prompt(self.svc(), req)
                off, _ = self.prompt(self.svc(0), req)
                self.assertEqual(on, off)
                self.assertEqual(log, "")

    def test_note_before_the_models_turn(self):
        req = agent(*[[call()]] * 4)
        before = copy.deepcopy(req)
        on, log = self.prompt(self.svc(), req)
        off, _ = self.prompt(self.svc(0), req)
        self.assertEqual(req, before)                                    # the client's conversation is not changed
        self.assertIn("[strata] loop guard: bash repeated 4 times", log)
        self.assertEqual(log.count("loop guard"), 1)
        note = LG.note("bash", 4)
        self.assertIn(note, on)
        head = off[:off.rindex("<|im_start|>assistant")]                 # everything up to the last tool result
        self.assertTrue(on.startswith(head))
        self.assertEqual(on[len(head):], "<|im_start|>user\n" + note + "<|im_end|>\n" + off[len(head):])

    def test_over_http(self):
        tok = ByteTokenizer()
        eng = MockEngine(tok, "ok", max_context=1 << 20)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                for api, path, body in (("openai", "/v1/chat/completions", agent(*[[call()]] * 5)),
                                        ("anthropic", "/v1/messages", None)):
                    if body is None:
                        msgs = [{"role": "user", "content": "go"}]
                        for k in range(4):
                            msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{k}", "name": "bash",
                                                                         "input": {"command": "make"}}]},
                                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{k}",
                                                                   "content": "error"}]}]
                        body = {"model": "x", "max_tokens": 20, "messages": msgs, "tools": [
                            {"name": "bash", "input_schema": {"type": "object", "properties": {}}}]}
                    req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                                 data=json.dumps(body).encode(), headers={
                                                     "Content-Type": "application/json",
                                                     "anthropic-version": "2023-06-01"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        r.read()
                    with self.subTest(api=api):
                        self.assertIn("[Note from the server, not the user", tok.decode(eng.last_prompt))
            log = out.getvalue()
            self.assertIn("[strata] loop guard: bash repeated 5 times", log)
            self.assertIn("[strata] loop guard: bash repeated 4 times", log)
        finally:
            httpd.shutdown()
            httpd.server_close()


class Ds41Template:
    """DeepSeek V4.1's encoder behind the server's template interface (as tools/v41/serve/serve_ds41.py renders it,
    without importing that module, which patches the server for good)."""

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        out = []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [{"type": "function", "id": None, "function": dict(c["function"])}
                                   for c in m["tool_calls"]]
            out.append(m)
        if tools:
            tools = [{"type": "function", "function": t} for t in tools]
            if out and out[0].get("role") == "system":
                out[0] = dict(out[0], tools=tools)
            else:
                out.insert(0, {"role": "system", "content": "", "tools": tools})
        return E.encode_messages(out, thinking_mode="chat" if kw.get("enable_thinking") is False else "thinking",
                                 reasoning_effort=kw.get("reasoning_effort"))


class DeepSeekPrompt(unittest.TestCase):
    """DeepSeek V4.1 takes a system message mid-conversation: the note is one, after the tool result, and the
    calls in the prompt stay DSML (Maya #844: the DSML case)."""

    def prompt(self, req, repeats=LG.REPEATS):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=1 << 20), tok, Ds41Template())
        svc.loop_guard_repeats = repeats
        messages, tools, kw = openai_to_messages(req)
        with contextlib.redirect_stdout(io.StringIO()):
            ids, _, _ = svc.prepare(messages, tools, kw, 10)
        return tok.decode(ids), svc

    def test_note_as_a_system_turn(self):
        dsml = call(args={"command": "cat log | tail -n 40", "timeout": 30})
        req = agent(*[[dsml]] * 4)
        on, svc = self.prompt(req)
        off, _ = self.prompt(req, 0)
        self.assertEqual(svc.note_role, "system")
        note = LG.note("bash", 4)
        self.assertEqual(on.count(note), 1)
        self.assertEqual(on.count(E.dsml_token + E.tool_calls_block_name), off.count(E.dsml_token + E.tool_calls_block_name))
        self.assertIn('string="false">30<', on)                       # a typed argument, as DSML writes it
        head = off[:off.rindex(E.ASSISTANT_SP_TOKEN)]
        self.assertTrue(on.startswith(head))
        self.assertEqual(on[len(head):], E.SYSTEM_SP_TOKEN + note + off[len(head):])
        self.assertTrue(on.endswith(E.ASSISTANT_SP_TOKEN + E.thinking_start_token))

    def test_no_repeats_render_as_before(self):
        for req in (agent(*[[call()]] * 3, result=changing),
                    agent([call()], [call(args={"command": "ls"})], [call()], [call()], result=changing)):
            with self.subTest(n=len(req["messages"])):
                self.assertEqual(self.prompt(req)[0], self.prompt(req, 0)[0])


class Windowed(unittest.TestCase):
    """0.95 (after Maya 770ddef / 989ee69): a call the last WINDOW calls since the user's message hold TIMES times with
    the same arguments and result - A, B, A, B loops and loops with other calls between, which no run in a row shows."""

    def w(self, req):
        return LG.windowed(template_messages(req))

    def test_alternating_calls(self):
        a, b = call("read", {"path": "a.py"}), call("read", {"path": "b.py"})
        self.assertIsNone(LG.repeated(template_messages(agent([a], [b], [a], [b], [a], [b]))))   # the run: never
        self.assertIsNone(self.w(agent([a], [b], [a], [b])))                                        # twice each
        self.assertEqual(self.w(agent([a], [b], [a], [b], [a])), ("read", 3))
        self.assertEqual(self.w(agent([a], [b], [a], [b], [a], [b])), ("read", 3))

    def test_three_in_a_row_with_the_same_result(self):
        self.assertEqual(self.w(agent(*[[call()]] * 3)), ("bash", 3))
        self.assertIsNone(self.w(agent(*[[call()]] * 2)))
        self.assertIsNone(self.w(agent(*[[call()]] * 3, result=changing)))       # a new result each time: no loop

    def test_numbers_only_in_a_row(self):
        """The same call right after itself, its result different only in numbers: the same (a check run again with
        nothing changed); with another call in between it needs the very same result."""
        self.assertEqual(self.w(agent(*[[call()]] * 3, result=lambda k, j: f"{k} failed, {k * 7} passed")), ("bash", 3))
        turns = [[call()], [call("read", {"path": "x"})]] * 2 + [[call()]]
        self.assertIsNone(self.w(agent(*turns, result=lambda k, j: f"{k} failed")))
        self.assertEqual(self.w(agent(*turns)), ("bash", 3))

    def test_non_ascii_results_are_compared(self):
        """Results in CJK text that differ are not the same; ones that differ only in numbers still are."""
        logs = ["日志：成功", "日志：失败", "错误：磁盘已满"]
        self.assertIsNone(self.w(agent(*[[call()]] * 3, result=lambda k, j: logs[k])))
        self.assertEqual(self.w(agent(*[[call()]] * 3, result=lambda k, j: f"日志：{k} 个失败")), ("bash", 3))

    def test_a_wait_needs_the_same_result(self):
        poll = call(args={"command": "sleep 30; tail -n 1 train.log"})
        self.assertIsNone(self.w(agent(*[[poll]] * 4, result=lambda k, j: f"step {k * 100}")))
        self.assertEqual(self.w(agent(*[[poll]] * 3, result="step 100")), ("bash", 3))

    def test_the_window_is_twelve_calls(self):
        a = call("read", {"path": "a.py"})
        other = [[call("read", {"path": f"f{k}.py"})] for k in range(10)]
        self.assertIsNone(self.w(agent([a], *other, [a], [a])))                 # the first one fell out
        self.assertEqual(self.w(agent([a], *other[:9], [a], [a])), ("read", 3))

    def test_the_users_message_starts_over(self):
        a, b = call("read", {"path": "a.py"}), call("read", {"path": "b.py"})
        first = template_messages(agent([a], [b], [a]))
        again = [{"role": "assistant", "content": "", "tool_calls": [dict(a, id="z")]},
                 {"role": "tool", "tool_call_id": "z", "content": "1 failed"}]
        for text, fires in (("now look at b.py instead", False), ("continue", True), ("OK.", True),
                            ("<system-reminder>todo list changed</system-reminder>", True),
                            ("<system-reminder>x</system-reminder> and also fix c.py", False)):
            with self.subTest(text=text):
                msgs = first + [{"role": "assistant", "content": "Done."}, {"role": "user", "content": text}] + again
                self.assertEqual(LG.windowed(msgs) is not None, fires)
        parts = [{"type": "text", "text": "go on"}]
        self.assertTrue(LG.no_news({"role": "user", "content": parts}))

    def test_only_the_windows_results_are_read(self):
        """A long session: only the last WINDOW results are serialised and gisted, not every result before them."""
        a, b = call("read", {"path": "a.py"}), call("read", {"path": "b.py"})
        msgs = template_messages(agent(*[[call("read", {"path": f"f{k}.py"})] for k in range(40)], [a], [b], [a], [b], [a]))
        with mock.patch.object(LG, "_gist", wraps=LG._gist) as gist:
            self.assertEqual(LG.windowed(msgs), ("read", 3))
        self.assertLessEqual(gist.call_count, LG.WINDOW)

    def test_one_turn_counts_once(self):
        """A batch with the same call three times is no loop by itself."""
        self.assertIsNone(self.w(agent([call(), call(), call()])))
        self.assertEqual(self.w(agent([call(), call()], [call("read", {"path": "x"})], [call()], [call()])), ("bash", 3))

    def test_only_the_last_turns_calls(self):
        """A loop that is over (the last turn made something else) gets no note."""
        self.assertIsNone(self.w(agent(*[[call()]] * 3, [call("read", {"path": "x"})])))

    def test_only_when_the_request_answers_the_call(self):
        req = agent(*[[call()]] * 3)
        req["messages"] = req["messages"][:-1]
        self.assertIsNone(self.w(req))
        self.assertIsNone(LG.windowed(template_messages(agent(*[[call()]] * 3)), 0))     # off
        self.assertIsNone(LG.windowed([]))

    def test_anthropic_alternating(self):
        msgs = [{"role": "user", "content": "fix it"}]
        for k in range(5):
            path = "a.py" if k % 2 == 0 else "b.py"
            msgs.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": f"t{k}", "name": "Read", "input": {"file_path": path}}]})
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"t{k}", "content": f"contents of {path}"},
                {"type": "text", "text": "<system-reminder>Remember the todo list.</system-reminder>"}]})
        self.assertEqual(LG.windowed(anthropic_to_messages({"messages": msgs, "max_tokens": 10})[0]), ("Read", 3))

    def test_setting(self):
        self.assertEqual(LG.window_setting(None), LG.TIMES)
        self.assertEqual(LG.window_setting({"repeats": 5}), LG.TIMES)
        self.assertEqual(LG.window_setting({"window": False}), 0)
        self.assertEqual(LG.window_setting(False), 0)
        self.assertEqual(LG.setting({"window": False}), LG.REPEATS)
        self.assertEqual(LG.setting({"repeats": 5, "window": True}), 5)
        with self.assertRaises(ValueError):
            LG.setting({"window": "no"})

    def test_prompt(self):
        """Through the server: the window's note after the last result; the run's note wins when both fire; a
        guard turned off (repeats 0) or the window off leaves the prompt as it was."""
        q = QwenPrompt()
        a, b = call("read", {"path": "a.py"}), call("read", {"path": "b.py"})
        req = agent([a], [b], [a], [b], [a])
        on, log = q.prompt(q.svc(), req)
        off, _ = q.prompt(q.svc(0), req)
        note = LG.window_note("read", 3)
        head = off[:off.rindex("<|im_start|>assistant")]
        self.assertEqual(on[len(head):], "<|im_start|>user\n" + note + "<|im_end|>\n" + off[len(head):])
        self.assertIn("[strata] loop guard: read made 3 times in the last 12 calls", log)
        svc = q.svc()
        svc.loop_guard_window = 0
        self.assertEqual(q.prompt(svc, req)[0], off)
        both, log = q.prompt(q.svc(), agent(*[[call()]] * 4))
        self.assertIn(LG.note("bash", 4), both)
        self.assertNotIn(LG.window_note("bash", 4), both)



class ReminderScan(unittest.TestCase):
    """strip_reminders gives REMINDER.search / REMINDER.sub's results in linear time (the regex was quadratic on many
    openers with no closer, and runs on every chat request)."""

    def test_same_results_as_the_regex(self):
        import random
        from serve import loop_guard as LG
        pieces = ["<system-reminder>", "</system-reminder>", "x", " ", "\n", "<system-", "reminder>", "</system"]
        rng = random.Random(7)
        cases = ["", "hi", "<system-reminder>a</system-reminder>", "<system-reminder>a</system-reminder> b",
                 "<system-reminder>a", "</system-reminder><system-reminder>b</system-reminder>",
                 "<system-reminder><system-reminder>a</system-reminder>x</system-reminder>"]
        cases += ["".join(rng.choice(pieces) for _ in range(rng.randrange(0, 12))) for _ in range(3000)]
        for text in cases:
            want = (LG.REMINDER.search(text) is not None, LG.REMINDER.sub("", text))
            self.assertEqual(LG.strip_reminders(text), want, repr(text))

    def test_many_openers_without_a_closer_are_fast(self):
        import time
        from serve import loop_guard as LG
        for text in ("<system-reminder>" * 20000, "<system-reminder>x" * 20000 + "</system-reminder>"):
            t0 = time.perf_counter()
            LG.no_news({"role": "user", "content": text})
            self.assertLess(time.perf_counter() - t0, 0.5)
        self.assertFalse(LG.no_news({"role": "user", "content": "<system-reminder>" * 20000}))
        self.assertTrue(LG.no_news({"role": "user", "content": "<system-reminder>x</system-reminder>  "}))

if __name__ == "__main__":
    unittest.main()
