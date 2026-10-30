"""Small CPU checks for the M31 common-server adapter (no model or GPU)."""
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools/m31/serve")]
from m31_tok import M31Tokenizer  # noqa: E402
import m31_parser as P  # noqa: E402
import serve_m31 as M  # noqa: E402
from serve import server as S  # noqa: E402
from serve import deep_sleep as D  # noqa: E402
import strata_tokenizer as ST  # noqa: E402


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokens = list(ST.BYTE_TO_UNICODE.values()) + ["[e~[", P.NS]
        self.tok = M31Tokenizer(self.tokens, [], [1] * 256 + [3, 4])

    def test_normalization_and_special_partition(self):
        self.assertEqual(self.tok.encode("e\u0301"), self.tok.encode("é"))
        self.assertEqual(self.tok.decode(self.tok.encode("e\u0301")), "é")
        self.assertEqual(self.tok.encode(P.NS), [257])
        self.assertEqual(self.tok.encode("[e~[", parse_special=True), [256])
        self.assertNotEqual(self.tok.encode("[e~["), [256])
        self.assertEqual(self.tok.token_bytes(257), P.NS.encode())

    def test_split_digits_and_unicode_roundtrip(self):
        self.assertEqual(self.tok._pre_split("1234567"), ["123", "456", "7"])
        for text in ("a漢字def", "Привет, мир!", "😀\r\n", "DON'T I'll", "\x00\x01"):
            self.assertEqual(self.tok.decode(self.tok.encode(text)), text)


TOOLS = [{"name": "lookup", "parameters": {"type": "object", "required": ["query", "options"], "properties": {
    "query": {"type": "string"},
    "options": {"type": "object", "properties": {"cities": {"type": "array", "items": {
        "type": "object", "properties": {"name": {"type": "string"}, "count": {"type": "integer"}}}},
        "enabled": {"type": "boolean"}}}}}}]


class ParserTests(unittest.TestCase):
    def test_optional_reasoning_and_every_stream_boundary(self):
        for text, expected in (("answer", [("content", "answer")]),
                               (P.THINK_END + "answer", [("content", "answer")]),
                               (P.THINK_START + " why" + P.THINK_END + "answer",
                                [("reasoning", "why"), ("content", "answer")])):
            for stride in range(1, len(text) + 1):
                parser, events = P.M31OutputParser(), []
                for i in range(0, len(text), stride):
                    events.extend(parser.feed(text[i:i + stride]))
                events.extend(parser.finish())
                actual = []
                for event in events:
                    if actual and actual[-1][0] == event.kind:
                        actual[-1] = (event.kind, actual[-1][1] + event.text)
                    else:
                        actual.append((event.kind, event.text))
                self.assertEqual(actual, expected, (text, stride))

    def test_recursive_unicode_tools_in_single_byte_deltas(self):
        tag = lambda name, value: P.NS + "<" + name + ">" + value + P.NS + "</" + name + ">"
        invoke = (P.NS + '<invoke name="lookup">' + tag("query", "Москва & <literal>") +
                  tag("options", tag("cities", tag("item", tag("name", "東京") + tag("count", "2"))) +
                      tag("enabled", "true")) + P.INVOKE_END)
        text = "before" + P.CALL_START + invoke + invoke + P.CALL_END + "after"
        parser, events = P.M31OutputParser(tools=TOOLS), []
        for char in text:
            events.extend(parser.feed(char))
        events.extend(parser.finish())
        calls = [event.call for event in events if event.kind == "tool_call"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].arguments, {"query": "Москва & <literal>", "options": {
            "cities": [{"name": "東京", "count": 2}], "enabled": True}})
        self.assertEqual("".join(event.text for event in events if event.kind == "content"), "beforeafter")

    def test_incomplete_tool_is_error(self):
        parser = P.M31OutputParser(tools=TOOLS)
        parser.feed(P.CALL_START + P.NS + '<invoke name="lookup">')
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parser.finish()


class TokenAwareRuntimeTests(unittest.TestCase):
    def setUp(self):
        tokens = list(ST.BYTE_TO_UNICODE.values()) + list(P.CONTROL_PIECES) + ["[e~["]
        self.tok = M31Tokenizer(tokens, [], [1] * 256 + [4] * len(P.CONTROL_PIECES) + [3])
        self.tools = [{"name": "lookup", "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}]

    def plain(self, text):
        # USER_DEFINED markers are always special, even with parse_special=False.
        # A plain span deliberately encodes their spelling as ordinary model IDs.
        return self.tok.encode(text, parse_special=False, plain=[(0, len(text))])

    def run_ids(self, ids):
        class Template:
            def render(self, *args, **kwargs):
                return "abc"
        engine = S.MockEngine(self.tok, "")
        engine.script = ids + self.tok.encode("[e~[", parse_special=True)
        engine.scripts = [engine.script]
        service = M.M31Service(engine, self.tok, Template())
        with mock.patch.object(S, "OutputParser", P.M31OutputParser), \
                mock.patch.object(S, "Detokenizer", M.M31Detokenizer):
            items = list(service.run([1], True, self.tools, 2000, {"temperature": 0}, threading.Event()))
        return [value for kind, value in items if kind == "event"]

    def test_actual_thinking_controls_differ_from_ordinary_spelling(self):
        text = P.THINK_START + "why😀" + P.THINK_END + "answer東京"
        special = self.tok.encode(text, parse_special=True)
        self.assertEqual(special, self.tok.encode(text, parse_special=False))
        self.assertNotEqual(special, self.plain(text))
        events = self.run_ids(special)
        self.assertEqual("".join(event.text for event in events if event.kind == "reasoning"), "why😀")
        self.assertEqual("".join(event.text for event in events if event.kind == "content"), "answer東京")
        ordinary = self.run_ids(self.plain(text))
        self.assertTrue(all(event.kind == "content" for event in ordinary))
        self.assertEqual("".join(event.text for event in ordinary), text)

    def test_actual_tool_controls_differ_from_ordinary_spelling(self):
        text = (P.CALL_START + P.NS + '<invoke name="lookup">' + P.NS + "<query>東京😀" +
                P.NS + "</query>" + P.INVOKE_END + P.CALL_END)
        special = self.run_ids(self.tok.encode(text, parse_special=True))
        self.assertEqual([event.call.arguments for event in special if event.kind == "tool_call"],
                         [{"query": "東京😀"}])
        ordinary = self.run_ids(self.plain(text))
        self.assertTrue(all(event.kind == "content" for event in ordinary))
        self.assertEqual("".join(event.text for event in ordinary), text)

    def test_ordinary_namespace_inside_string_cannot_close_argument(self):
        prefix = P.CALL_START + P.NS + '<invoke name="lookup">' + P.NS + "<query>"
        literal = "Москва😀 " + P.NS + "</query> " + P.THINK_END + " & <literal>"
        suffix = P.NS + "</query>" + P.INVOKE_END + P.CALL_END
        ids = self.tok.encode(prefix, parse_special=True) + self.plain(literal) + self.tok.encode(suffix, parse_special=True)
        events = self.run_ids(ids)
        self.assertEqual([event.call.arguments for event in events if event.kind == "tool_call"],
                         [{"query": literal}])

    def test_ordinary_thinking_terminator_stays_inside_reasoning(self):
        literal = "quote " + P.THINK_END + " and 😀"
        ids = (self.tok.encode(P.THINK_START, parse_special=True) + self.plain(literal) +
               self.tok.encode(P.THINK_END + "answer", parse_special=True))
        events = self.run_ids(ids)
        self.assertEqual("".join(event.text for event in events if event.kind == "reasoning"), literal)
        self.assertEqual("".join(event.text for event in events if event.kind == "content"), "answer")


class AdapterTests(unittest.TestCase):
    def test_cache_and_native_grammar_metadata(self):
        key = M.M31Engine.sampling_keys({"cache_prompt": False, "m31_chat_b64": M.chat_key({"a": "é"})})
        self.assertIn(" cache_prompt=0", key)
        encoded = key.split(" chat_b64=")[1]
        self.assertEqual(json.loads(base64.urlsafe_b64decode(encoded)), {"a": "é"})
        with self.assertRaisesRegex(ValueError, "boolean"):
            M.M31Engine.sampling_keys({"cache_prompt": 0})

    def test_native_top_k_and_full_uint64_seed_are_preserved(self):
        for top_k in (0, 40, 128):
            for seed in (0, 1, 2**64 - 1):
                key = M.M31Engine.sampling_keys({"top_k": top_k, "seed": seed})
                self.assertIn(f" top_k={top_k}", key)
                self.assertIn(f" seed={seed}", key)
        self.assertNotIn(" seed=", M.M31Engine.sampling_keys({"seed": -1}))
        with self.assertRaisesRegex(ValueError, "uint64"):
            M.M31Engine.sampling_keys({"seed": 2**64})
        self.assertEqual(M.sampling_defaults_from_config({"sampling": {"top_k": 0, "seed": 0}}),
                         {"top_k": 0, "seed": 0})

    def test_invalid_sampling_is_not_silently_turned_into_greedy(self):
        for values in ({"temperature": float("nan")}, {"temperature": -1}, {"top_p": 0},
                       {"min_p": float("inf")}, {"top_k": -1}, {"seed": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                M.M31Engine.sampling_keys(values)
        with self.assertRaises(ValueError):
            M.sampling_defaults_from_config({"sampling": {"temperature": float("nan")}})
        self.assertIn(" penalty_last_n=0", M.M31Engine.sampling_keys({"penalty_last_n": 0,
                                                                     "repetition_penalty": 1.1}))

    def test_fusion_guard_runs_before_process_start(self):
        with self.assertRaisesRegex(ValueError, "FUSION=1"):
            M.M31Engine("missing", [], env={"GGML_CUDA_DISABLE_FUSION": "0"}, lazy=True)
        with mock.patch.dict("os.environ", {"GGML_CUDA_DISABLE_FUSION": "1"}):
            engine = M.M31Engine("missing", [], lazy=True)
        self.assertEqual(engine.spawn[4]["GGML_CUDA_DISABLE_FUSION"], "1")

    def test_template_calls_are_objects(self):
        messages = [{"role": "assistant", "tool_calls": [{"name": "lookup", "arguments": '{"q":"東京"}'}]}]
        converted = M.template_messages(messages)
        self.assertEqual(converted[0]["tool_calls"][0]["function"]["arguments"], {"q": "東京"})
        self.assertIsInstance(messages[0]["tool_calls"][0]["arguments"], str)

    def test_tools_match_native_common_canonical_order_and_defaults(self):
        tools = [{"parameters": {"type": "object"}, "strict": True, "name": "lookup"}]
        canonical = M.openai_tools(tools)
        self.assertEqual(canonical, [{"type": "function", "function": {"name": "lookup", "description": "",
                                                                       "parameters": {"type": "object"}}}])
        self.assertEqual(list(canonical[0]["function"]), ["name", "description", "parameters"])

    def test_fit_max_tokens_and_cache_flag_survive_prepare(self):
        class Template:
            def render(self, *args, **kwargs):
                return "abc"
        engine = S.MockEngine(S.ByteTokenizer(), "ok", max_context=128)
        service = M.M31Service(engine, S.ByteTokenizer(), Template(), fit_max_tokens=False)
        ids, _, maximum = service.prepare([{"role": "user", "content": "hi"}], [], {}, 200,
                                          req={"cache_prompt": False})
        self.assertTrue(service.fit_max_tokens)
        self.assertEqual(service.sampling_defaults, {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.05})
        self.assertEqual(maximum, 128 - S.CTX_SLACK - 3)
        self.assertFalse(ids.cache_prompt)
        self.assertEqual(ids.chat_request["messages"], [{"role": "user", "content": "hi"}])

    def test_unsupported_config_fails_before_process_creation(self):
        for config in ({"vision": {"exe": "unavailable"}}, {"parallel": 2}, {"engine_silence_s": 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                M.validate_config(config)

    def test_named_tool_choice_is_rendered_as_one_required_tool(self):
        class Template:
            def render(self, messages, tools=None, **kwargs):
                self.tools = tools
                return "abc"
        template = Template()
        service = M.M31Service(S.MockEngine(S.ByteTokenizer(), "ok"), S.ByteTokenizer(), template)
        ids, _, _ = service.prepare([], TOOLS + [{"name": "other", "parameters": {}}], {},
                                   req={"tool_choice": {"type": "function", "function": {"name": "lookup"}}})
        self.assertEqual([tool["name"] for tool in template.tools], ["lookup"])
        self.assertEqual(ids.chat_request["tool_choice"], "required")
        self.assertEqual(ids.chat_request["tools"][0]["function"]["name"], "lookup")


@unittest.skipIf(os.name == "nt", "the fake executable uses a POSIX shebang")
class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        script = self.script = Path(self.tmp.name) / "fake_engine.py"
        script.write_text("#!" + sys.executable + "\n" + '''
import sys
from pathlib import Path
counter = 0
used = Path(__file__).with_suffix(".used")
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
    if line.startswith("STOP"):
        if not Path(__file__).with_suffix(".hang").exists():
            print("DONE 0 1 1 1 cancel 0 0 0 0 0", flush=True)
    elif line.startswith("GEN"):
        counter += 1
        if not used.exists():
            used.touch()
            text = ']<]minimax[>[<tool_call>]<]minimax[>[<invoke name="unknown">'
            for char in text:
                print("T", ord(char), flush=True)
        else:
            print("T 111\\nT 107\\nDONE 2 1 1 1 stop 0 0 0 0 0", flush=True)
''', encoding="utf-8")
        script.chmod(0o700)
        self.engine = M.M31Engine(str(script), [])
        self.engine.silence_s = 0.5
        self.addCleanup(self.engine.close)

    def test_parser_exception_drains_stop_and_releases_fifo(self):
        class Template:
            def render(self, *args, **kwargs):
                return "abc"
        service = M.M31Service(self.engine, S.ByteTokenizer(), Template())
        service.stop_ids = set()
        with mock.patch.object(S, "OutputParser", P.M31OutputParser):
            with self.assertRaisesRegex(ValueError, "unknown MiniMax tool"):
                list(service.run([1], False, TOOLS, 100, {}, threading.Event()))
            self.assertFalse(service.fifo.locked())
            events = list(service.run([1], False, [], 100, {}, threading.Event()))
        self.assertTrue(self.engine.alive())
        self.assertEqual("".join(value.text for kind, value in events if kind == "event"), "ok")

    def test_cache_flag_reaches_real_common_protocol_pipe(self):
        captured = []
        original = self.engine.proc.stdin.write
        def write(text):
            captured.append(text)
            return original(text)
        ids = M.PreparedIds([1], {"messages": [{"role": "user", "content": "hi"}]}, False)
        # The fake's first request is an unfinished malformed call: close() must
        # send STOP and consume its DONE before a following request can run.
        with mock.patch.object(self.engine.proc.stdin, "write", side_effect=write):
            gen = self.engine.generate(ids, 100, {}, threading.Event())
            next(gen)
            gen.close()
        self.assertIn("cache_prompt=0", captured[0])
        self.assertIn("chat_b64=", captured[0])
        self.assertEqual(captured[-1], "STOP\n")
        self.assertEqual(list(self.engine.generate([1], 100, {}, threading.Event())), [111, 107])

    def test_unacknowledged_stop_has_deadline_then_next_request_restarts(self):
        class Template:
            def render(self, *args, **kwargs):
                return "abc"
        service = M.M31Service(self.engine, S.ByteTokenizer(), Template())
        service.stop_ids = set()
        self.script.with_suffix(".hang").touch()
        started = time.monotonic()
        with mock.patch.object(S, "OutputParser", P.M31OutputParser):
            with self.assertRaises(S.EngineSilent):
                list(service.run([1], False, TOOLS, 100, {}, threading.Event()))
            self.assertLess(time.monotonic() - started, 3)
            self.assertFalse(service.fifo.locked())
            self.assertFalse(self.engine.alive())
            events = list(service.run([1], False, [], 100, {}, threading.Event()))
        self.assertTrue(self.engine.alive())
        self.assertEqual("".join(value.text for kind, value in events if kind == "event"), "ok")


class SleepProtocolTests(unittest.TestCase):
    """SLEEP/WAKE through the real common pipe with a fake engine process."""
    def engine(self, capable=True):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / "fake_engine.py"
        script.write_text("#!" + sys.executable + "\n" + f'''
import sys
from pathlib import Path
here = Path(__file__)
print("INFO prefix_cache=1 sleep={int(capable)}", flush=True)
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    with here.with_suffix(".log").open("a") as log:
        log.write(line.split()[0] + "\\n")
    if line.startswith("QUIT"):
        break
    if line.startswith("SLEEP"):
        print("SLEPT seconds=0.5 kv_tokens=3 kv_bytes=96 dense_bytes=10 experts=2 conversation=1 context=1", flush=True)
    elif line.startswith("WAKE"):
        if here.with_suffix(".refuse").exists():
            print("ERR wake failed (still asleep): out of memory", flush=True)
        else:
            print("INFO woke=1 wake_seconds=1.25 kv_tokens=3", flush=True)
            print("READY 4096 stop", flush=True)
    elif line.startswith("GEN"):
        print("T 111\\nT 107\\nDONE 2 1 1 1 stop 0 0 0 0 0", flush=True)
''', encoding="utf-8")
        script.chmod(0o700)
        patch = mock.patch.object(D, "driver", lambda: None)   # no CUDA checkpoint fallback: native sleep only
        patch.start()
        self.addCleanup(patch.stop)
        engine = M.M31Engine(str(script), [])
        engine.silence_s = 5
        self.addCleanup(engine.close)
        return engine, script

    def test_sleep_figures_then_generation_wakes_first(self):
        engine, script = self.engine()
        figures = engine.sleep()
        self.assertTrue(engine.sleeping)
        self.assertEqual(figures["kv_tokens"], 3)
        self.assertEqual(figures["seconds"], 0.5)
        self.assertEqual(list(engine.generate([1], 4, {}, threading.Event())), [111, 107])
        self.assertFalse(engine.sleeping)
        self.assertEqual(engine.info["wake"]["wake_seconds"], 1.25)
        self.assertEqual(script.with_suffix(".log").read_text().split(), ["SLEEP", "WAKE", "GEN"])

    def test_failed_wake_is_gpu_busy_and_stays_asleep(self):
        engine, script = self.engine()
        engine.sleep()
        script.with_suffix(".refuse").touch()
        with self.assertRaises(S.GpuBusy):
            engine.wake()
        self.assertTrue(engine.sleeping)
        script.with_suffix(".refuse").unlink()
        self.assertEqual(engine.wake()["kv_tokens"], 3)
        self.assertFalse(engine.sleeping)

    def test_engine_without_sleep_support_is_refused_before_any_command(self):
        engine, script = self.engine(capable=False)
        with self.assertRaisesRegex(ValueError, "cannot sleep"):
            engine.sleep()
        self.assertFalse(script.with_suffix(".log").exists())


class HttpSleepTests(unittest.TestCase):
    def setUp(self):
        class Engine(S.MockEngine):
            sleeping, sleeps, wakes = False, 0, 0
            info = {}
            def alive(self):
                return True
            def sleep(self):
                self.sleeping, self.sleeps = True, self.sleeps + 1
                return {"seconds": 0.5, "kv_tokens": 3}
            def wake(self):
                self.sleeping, self.wakes = False, self.wakes + 1
                self.info["wake"] = {"wake_seconds": 1.25}
                return self.info["wake"]
            def generate(self, ids, maximum, sampling, cancel, embeddings=None):
                assert not self.sleeping, "generation reached a sleeping engine"
                yield from super().generate(ids, maximum, sampling, cancel, embeddings)
        class Template:
            def render(self, *args, **kwargs):
                return "abc"
        for patch in (mock.patch.object(S, "OutputParser", P.M31OutputParser),):
            patch.start()
            self.addCleanup(patch.stop)
        self.engine = Engine(S.ByteTokenizer(), "ok", max_context=256)
        self.service = M.M31Service(self.engine, S.ByteTokenizer(), Template())
        self.service.free_vram_mib = lambda: None
        self.httpd = S.serve(self.service, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def call(self, path, body=None, content_type="application/json"):
        request = urllib.request.Request(self.base + path, data=json.dumps(body or {}).encode(),
                                         headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def test_sleep_then_request_wakes_and_answers(self):
        self.assertEqual(self.call("/sleep"), (200, {"status": "asleep", "seconds": 0.5, "kv_tokens": 3}))
        self.assertTrue(self.service.v1_status()["asleep"])
        status, body = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, body)
        self.assertEqual((self.engine.sleeps, self.engine.wakes), (1, 1))
        self.assertFalse(self.service.v1_status()["asleep"])

    def test_wake_endpoint_and_states(self):
        self.assertEqual(self.call("/wake"), (200, {"status": "not asleep"}))
        self.call("/sleep")
        self.assertEqual(self.call("/wake"), (200, {"status": "awake", "wake_seconds": 1.25}))
        self.assertEqual(self.engine.wakes, 1)

    def test_control_rules_match_unload(self):
        self.assertEqual(self.call("/sleep", content_type="application/x-www-form-urlencoded")[0], 415)
        self.assertEqual(self.engine.sleeps, 0)
        with self.service.fifo:
            self.assertEqual(self.call("/sleep"), (409, {"status": "busy"}))
        self.assertEqual(self.engine.sleeps, 0)

    def test_engine_without_sleep_is_unsupported(self):
        del type(self.engine).sleep
        self.assertEqual(self.call("/sleep"), (501, {"status": "unsupported"}))


class HttpTests(unittest.TestCase):
    def setUp(self):
        class Tokenizer(S.ByteTokenizer):
            SPECIALS = S.ByteTokenizer.SPECIALS + ["[e~["]
        class Engine(S.MockEngine):
            info = {"logit_bias": 1}
            def generate(self, ids, maximum, sampling, cancel, embeddings=None):
                self.request_sampling, self.maximum = dict(sampling), maximum
                yield from super().generate(ids, maximum, sampling, cancel, embeddings)
        class Template:
            def render(self, *args, **kwargs):
                return "abc"
        self.patches = [mock.patch.object(S, "IM_END", "[e~["),
                        mock.patch.object(S, "OutputParser", P.M31OutputParser)]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.engine = Engine(Tokenizer(), '{"answer":"東京"}', max_context=256)
        self.service = M.M31Service(self.engine, Tokenizer(), Template())
        self.httpd = S.serve(self.service, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, **kwargs):
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps({
            "messages": [{"role": "user", "content": "hi"}], **kwargs}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def test_http_cache_prompt_and_bias_reach_generation(self):
        status, body = self.post(cache_prompt=False, logit_bias={"65": -2.5})
        self.assertEqual(status, 200, body)
        self.assertFalse(self.engine.request_sampling["cache_prompt"])
        self.assertIn("chat_b64=", M.M31Engine.sampling_keys(self.engine.request_sampling))
        self.assertIn(" logit_bias=65:-2.5", M.M31Engine.sampling_keys(self.engine.request_sampling))
        status, body = self.post(cache_prompt="false")
        self.assertEqual(status, 400, body)
        self.assertFalse(self.service.fifo.locked())

    def test_ignore_eos_is_request_local(self):
        stops = set(self.service.stop_ids)
        status, body = self.post(ignore_eos=True)
        self.assertEqual(status, 200, body)
        self.assertTrue(self.engine.request_sampling["ignore_eos"])
        self.assertIn(" ignore_eos=1", M.M31Engine.sampling_keys(self.engine.request_sampling))
        self.assertEqual(self.service.stop_ids, stops)
        observed = []
        self.service._m31_request.ignore_eos = True
        worker = threading.Thread(target=lambda: observed.append(self.service.stop_ids))
        worker.start()
        worker.join()
        self.assertEqual(observed, [stops])
        self.service._m31_request.ignore_eos = False

    def test_http_max_completion_tokens_and_context_clipping(self):
        for values, maximum in (({}, 256 - S.CTX_SLACK - 3),
                                ({"max_tokens": 10000}, 256 - S.CTX_SLACK - 3),
                                ({"max_tokens": 10000, "max_completion_tokens": 40}, 40)):
            status, body = self.post(**values)
            self.assertEqual(status, 200, body)
            self.assertEqual(self.engine.maximum, maximum)

    def test_http_explicit_zero_never_calls_engine_for_all_aliases(self):
        for values in ({"max_tokens": 0}, {"max_completion_tokens": 0}, {"n_predict": 0},
                       {"max_tokens": 100, "max_completion_tokens": 0},
                       {"max_tokens": 100, "n_predict": 0}):
            with self.subTest(values=values), mock.patch.object(self.engine, "generate") as generate:
                status, body = self.post(**values)
                self.assertEqual(status, 200, body)
                generate.assert_not_called()
                self.assertEqual(body["usage"]["completion_tokens"], 0)
                self.assertIn(body["choices"][0]["message"]["content"], (None, ""))
                self.assertEqual(body["choices"][0]["finish_reason"], "length")

    def test_http_minus_one_uses_remaining_context_for_all_aliases(self):
        for key in ("max_tokens", "max_completion_tokens", "n_predict"):
            with self.subTest(key=key):
                status, body = self.post(**{key: -1})
                self.assertEqual(status, 200, body)
                self.assertEqual(self.engine.maximum, 256 - S.CTX_SLACK - 3)

    def test_http_boolean_and_invalid_limits_are_rejected(self):
        for key in ("max_tokens", "max_completion_tokens", "n_predict"):
            for value in (False, True, -2, 1.5):
                with self.subTest(key=key, value=value), mock.patch.object(self.engine, "generate") as generate:
                    status, body = self.post(**{key: value})
                    self.assertEqual(status, 400, body)
                    self.assertIn("integer", body["error"]["message"])
                    generate.assert_not_called()

    def test_http_oversize_limit_clips_at_one_remaining_token(self):
        with mock.patch.object(self.service, "encode_prompt", return_value=[1] * (256 - S.CTX_SLACK - 1)):
            status, body = self.post(max_tokens=10000)
        self.assertEqual(status, 200, body)
        self.assertEqual(self.engine.maximum, 1)
        self.assertEqual(body["usage"]["completion_tokens"], 1)

    def test_structured_output_passes_schema_to_native_worker(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
        status, body = self.post(response_format={"type": "json_schema", "json_schema": {
            "name": "answer", "schema": schema}})
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body["choices"][0]["message"]["content"]), {"answer": "東京"})
        native = json.loads(base64.urlsafe_b64decode(self.engine.request_sampling["m31_chat_b64"]))
        self.assertEqual(native["json_schema"], schema)


if __name__ == "__main__":
    unittest.main()
