"""serve/test_request_log.py - the opt-in request log (--request-log): one JSON line per request, against mock engines
(no GPU, no pack).

    python -m unittest serve.test_request_log -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import request_log as reqlog  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 8192
TEMPLATE = ROOT / "serve/chat_template.jinja"
SECRET_PROMPT = "SECRET-PROMPT-7f3a"
SECRET_ANSWER = "SECRET-ANSWER-91c2"
SECRET_KEY = "sk-SECRET-KEY-55aa"
TOOL = {"type": "function", "function": {"name": "search", "description": "look it up",
                                         "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
CALL = "<tool_call>\n<function=search>\n<parameter=q>\n2+2\n</parameter>\n</function>\n</tool_call>"


class ClockedEngine(MockEngine):
    """The mock engine with StrataEngine's clock: `last` as a DONE line gives it, the conversation cache holding the
    first `reused` tokens of every prompt (all but the last 20 when it is None)."""

    def __init__(self, *a, reused=5, **kw):
        super().__init__(*a, **kw)
        self.reused = reused

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t
        finally:
            reused = max(0, len(ids) - 20) if self.reused is None else min(self.reused, len(ids))
            self.last = {"generated": n, "prompt_tokens": len(ids), "prompt_ms": 40.0, "decode_ms": 20.0 * max(1, n),
                         "finish": "stop", "reused": reused, "hits": 9, "lookups": 10,
                         "prompt_read": len(ids) - reused}


class SlowEngine(MockEngine):
    """Writes one token, then waits (stopping on cancel): a client can hang up while it runs."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.cancelled = False
        yield from self.tok.encode("</think>\n\n")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 20:
            if cancel.is_set():
                self.cancelled = True
                return
            time.sleep(0.05)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class FakeVision:
    """An image encoder with a cache by the picture's bytes and a count of the pictures it encoded."""

    def __init__(self, d):
        self.dir = Path(d)
        self.cache, self.encodes = {}, 0

    def encode(self, source):
        key = str(source)
        if key not in self.cache:
            self.encodes += 1
            out = self.dir / f"{len(self.cache):016x}.sve"
            out.write_bytes(b"rows")
            self.cache[key] = (out, 3)
        return self.cache[key]


class Server:
    def __init__(self, engine, log_path=None, vision=None, monitor=False, detail=False, max_mb=64, api_key=""):
        self.tok = engine.tok
        self.svc = Service(engine, self.tok, ChatTemplate(TEMPLATE), vision=vision)
        self.svc.api_monitor = monitor
        self.svc.api_key = api_key
        if log_path is not None:
            self.svc.request_log = reqlog.RequestLog(str(log_path), max_mb=max_mb, detail=detail)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.svc.request_log is not None:
            self.svc.request_log.close()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "User-Agent": "reqlog-test/1.0",
                                              "anthropic-version": "2023-06-01", **(headers or {})})
        try:
            r = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            r = e
        with r:
            return r.status, r.read().decode()


def lines(path, want, timeout=10.0):
    """The log's lines once it has `want` of them (the line is written just after the answer is sent)."""
    t0 = time.monotonic()
    while True:
        text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
        rows = [json.loads(x) for x in text.splitlines()]
        if len(rows) >= want or time.monotonic() - t0 > timeout:
            return rows
        time.sleep(0.02)


def chat(content=SECRET_PROMPT, **kw):
    return {"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": content}], **kw}


class Off(unittest.TestCase):
    """Off by default: no record, no file, and the server window's done line as before."""

    def test_nothing_is_written_or_kept(self):
        with tempfile.TemporaryDirectory() as d:
            srv = Server(ClockedEngine(ByteTokenizer(), "</think>\n\nok", max_context=CTX))
            out = io.StringIO()
            try:
                self.assertIsNone(srv.svc.request_log)
                self.assertIsNone(srv.svc.begin_request("/v1/chat/completions", chat()))
                with contextlib.redirect_stdout(out):
                    self.assertEqual(srv.post("/v1/chat/completions", chat())[0], 200)
                    time.sleep(0.2)
            finally:
                srv.close()
            self.assertEqual(os.listdir(d), [])
            done = [x for x in out.getvalue().splitlines() if x.startswith("[strata] done:")]
            self.assertEqual(len(done), 1)
            self.assertRegex(done[0], r"^\[strata\] done: \d+ tokens in \d+ s \([\d.]+ tok/s\) \(stop, cancel=False\)"
                                      r", expert cache 90\.0% hit$")

    def test_bad_settings_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for kw in ({"max_mb": -1}, {"max_mb": "8"}, {"backups": 0}, {"backups": 1.5}):
                with self.subTest(**kw), self.assertRaises(ValueError):
                    reqlog.RequestLog(os.path.join(d, "r.jsonl"), **kw)
            with self.assertRaises(ValueError):
                reqlog.RequestLog("")


class On(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "logs" / "requests.jsonl"     # the folder is made

    def tearDown(self):
        self.dir.cleanup()

    def server(self, engine, **kw):
        srv = Server(engine, self.path, **kw)
        self.addCleanup(srv.close)
        return srv

    def test_one_line_per_request_with_its_figures(self):
        tok = ByteTokenizer()
        srv = self.server(ClockedEngine(tok, "Thought.</think>\n\n" + SECRET_ANSWER, max_context=CTX),
                          api_key=SECRET_KEY)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code, body = srv.post("/v1/chat/completions", chat(reasoning_effort="low"),
                                  {"Authorization": "Bearer " + SECRET_KEY})
        self.assertEqual(code, 200)
        reply = json.loads(body)
        (row,) = lines(self.path, 1)
        self.assertRegex(row["time"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
        self.assertRegex(row["end"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
        self.assertLessEqual(row["time"], row["end"])
        self.assertEqual((row["endpoint"], row["api"], row["model"], row["stream"]),
                         ("/v1/chat/completions", "openai", "m", False))
        self.assertEqual((row["client"], row["user_agent"]), ("127.0.0.1", "reqlog-test/1.0"))
        self.assertEqual((row["status"], row["outcome"], row["finish"], row["cancelled"]),
                         (200, "completed", "stop", False))
        self.assertEqual(row["queued_at_arrival"], 0)
        for key in ("queue_s", "start_s", "ttft_s", "total_s"):
            self.assertIsInstance(row[key], (int, float), key)
        self.assertLessEqual(row["start_s"], row["ttft_s"])
        self.assertLessEqual(row["ttft_s"], row["total_s"])
        usage = reply["usage"]
        self.assertEqual(row["prompt_tokens"], usage["prompt_tokens"])
        self.assertEqual(row["cached_tokens"], 5)
        self.assertEqual(row["read_tokens"], usage["prompt_tokens"] - 5)
        self.assertEqual(row["output_tokens"], usage["completion_tokens"])
        self.assertEqual(row["reasoning_tokens"], len("Thought.</think>"))     # one token a byte, the tag included
        self.assertEqual(row["tool_calls"], 0)
        self.assertEqual(row["prefill_ms"], 40.0)
        self.assertEqual(row["prefill_tok_s"], round(row["read_tokens"] / 0.04, 1))
        self.assertEqual(row["decode_tok_s"], 50.0)                       # 20 ms a token
        self.assertTrue(row["thinking"])
        self.assertEqual(row["max_tokens"], 64)
        self.assertNotIn("segments", row)                                 # detail is off
        self.assertNotIn("images", row)
        # the server window's done line carries the same time base and the request's id
        done = [x for x in out.getvalue().splitlines() if x.startswith("[strata] done:")]
        self.assertRegex(done[0], r", expert cache 90\.0% hit at \d{4}-\d\d-\d\dT[\d:.]+Z, request " + row["id"] + "$")
        raw = self.path.read_text(encoding="utf-8")
        for secret in (SECRET_PROMPT, SECRET_ANSWER, SECRET_KEY, "Thought.", "Bearer"):
            self.assertNotIn(secret, raw)

    def test_every_dialect_streamed_and_not(self):
        tok = ByteTokenizer()
        srv = self.server(ClockedEngine(tok, "</think>\n\n" + SECRET_ANSWER, max_context=CTX))
        sent = []
        for path, api in (("/v1/chat/completions", "openai"), ("/v1/messages", "anthropic"),
                          ("/v1/responses", "responses")):
            for stream in (False, True):
                body = ({"model": "m", "max_output_tokens": 64, "input": SECRET_PROMPT, "stream": stream}
                        if api == "responses" else chat(stream=stream))
                self.assertEqual(srv.post(path, body)[0], 200, (path, stream))
                sent.append((path, api, stream))
                lines(self.path, len(sent))
        rows = lines(self.path, len(sent))
        self.assertEqual([(r["endpoint"], r["api"], r["stream"]) for r in rows], sent)
        for r in rows:
            self.assertEqual((r["status"], r["outcome"], r["finish"]), (200, "completed", "stop"), r)
            self.assertGreater(r["output_tokens"], 0)
            self.assertIn("ttft_s", r)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn(SECRET_PROMPT, raw)
        self.assertNotIn(SECRET_ANSWER, raw)

    def test_conversation_cache_reuse(self):
        tok = ByteTokenizer()
        srv = self.server(ClockedEngine(tok, "</think>\n\nok", max_context=CTX, reused=None))
        self.assertEqual(srv.post("/v1/chat/completions", chat("a question " * 20))[0], 200)
        (row,) = lines(self.path, 1)
        self.assertEqual(row["cached_tokens"], row["prompt_tokens"] - 20)
        self.assertEqual(row["read_tokens"], 20)
        self.assertEqual(row["prefill_tok_s"], 500.0)                     # 20 tokens in 40 ms

    def test_tool_calls_are_counted(self):
        tok = ByteTokenizer()
        srv = self.server(ClockedEngine(tok, "</think>\n\n" + CALL, max_context=CTX))
        code, body = srv.post("/v1/chat/completions", chat(tools=[TOOL], max_tokens=256))
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["choices"][0]["finish_reason"], "tool_calls")
        (row,) = lines(self.path, 1)
        self.assertEqual(row["tool_calls"], 1)
        self.assertNotIn("2+2", self.path.read_text(encoding="utf-8"))   # nor the call's arguments

    def test_stop_sequence(self):
        tok = ByteTokenizer()
        srv = self.server(ClockedEngine(tok, "</think>\n\none two three", max_context=CTX))
        self.assertEqual(srv.post("/v1/chat/completions", chat(stop=["two"]))[0], 200)
        (row,) = lines(self.path, 1)
        self.assertEqual((row["finish"], row["stop_sequence"]), ("stop", True))
        self.assertNotIn('"two"', self.path.read_text(encoding="utf-8"))

    def test_client_that_hangs_up(self):
        tok = ByteTokenizer()
        engine = SlowEngine(tok, "OK", max_context=CTX)
        srv = self.server(engine)
        for stream in (False, True):
            with self.subTest(stream=stream):
                body = json.dumps(chat(stream=stream)).encode()
                c = socket.create_connection(("127.0.0.1", srv.port))
                c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json"
                          b"\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                time.sleep(1.0)
                c.close()
                rows = lines(self.path, 1 + stream)
                self.assertEqual(len(rows), 1 + stream)
                row = rows[-1]
                self.assertTrue(row["cancelled"], row)
                self.assertIn(row["finish"], ("cancel", "disconnect"))
                self.assertEqual(row["stream"], stream)
                self.assertTrue(engine.cancelled)

    def test_errors(self):
        tok = ByteTokenizer()
        engine = ClockedEngine(tok, "</think>\n\nok", max_context=CTX)
        srv = self.server(engine)
        self.assertEqual(srv.post("/v1/chat/completions", chat(max_tokens=CTX * 2))[0], 400)
        row = lines(self.path, 1)[-1]
        self.assertEqual((row["status"], row["outcome"], row["error_type"]), (400, "error", "invalid_request_error"))
        self.assertIn("context window", row["error"])
        self.assertNotIn("output_tokens", row)                           # it never ran
        def died(*a, **kw):
            raise EngineDied("stopped")
            yield                                                         # a generator, as the engine's

        with mock.patch.object(engine, "generate", side_effect=died):
            self.assertEqual(srv.post("/v1/chat/completions", chat())[0], 503)
            row = lines(self.path, 2)[-1]
            self.assertEqual((row["status"], row["outcome"]), (503, "error"))
            self.assertEqual(row["finish"], "error")
            self.assertIn("stopped", row["error"])
            self.assertEqual(srv.post("/v1/chat/completions", chat(stream=True))[0], 200)   # the stream had begun
            row = lines(self.path, 3)[-1]
            self.assertEqual((row["status"], row["outcome"], row["stream"]), (200, "error", True))

    def test_pictures(self):
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as vd:
            engine = ClockedEngine(tok, "</think>\n\nok", max_context=CTX, reused=0)
            srv = self.server(engine, vision=FakeVision(vd))

            def turn(*images):
                content = [{"type": "text", "text": SECRET_PROMPT}] + [
                    {"type": "image_url", "image_url": {"url": src}} for src in images]
                self.assertEqual(srv.post("/v1/chat/completions", chat(content))[0], 200)

            turn(os.path.join(vd, "a.png"), os.path.join(vd, "b.png"))
            row = lines(self.path, 1)[-1]
            self.assertEqual((row["images"], row["images_encoded"], row["images_from_encoder_cache"]), (2, 2, 0))
            self.assertEqual(row["images_in_cached_prefix"], 0)              # nothing was reused
            self.assertGreaterEqual(row["image_wait_s"], 0)
            self.assertGreaterEqual(row["image_encode_s"], 0)
            engine.reused = None                                          # all but the last 20 tokens: both pictures
            turn(os.path.join(vd, "a.png"), os.path.join(vd, "b.png"), os.path.join(vd, "c.png"))
            row = lines(self.path, 2)[-1]
            self.assertEqual((row["images"], row["images_encoded"], row["images_from_encoder_cache"]), (3, 1, 2))
            self.assertEqual(row["images_in_cached_prefix"], 3)
            engine.reused = 5                                                 # the template's start only
            turn(os.path.join(vd, "a.png"))
            row = lines(self.path, 3)[-1]
            self.assertEqual((row["images"], row["images_encoded"], row["images_in_cached_prefix"]), (1, 0, 0))
            self.assertNotIn(vd, self.path.read_text(encoding="utf-8"))      # no picture paths

    def test_detail_has_the_engine_figures(self):
        tok = ByteTokenizer()

        class DoneLine(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                try:
                    yield from super().generate(ids, max_new, sampling, cancel, embeddings)
                finally:
                    StrataEngine._parse_done(self, f"DONE 4 {len(ids)} 40.0 30.0 stop 3 5 7 9 10 0 0 0.0 "
                                                   f"{len(ids) - 7} 0")

        srv = self.server(DoneLine(tok, "</think>\n\nok", max_context=CTX), detail=True)
        self.assertEqual(srv.post("/v1/chat/completions", chat())[0], 200)
        (row,) = lines(self.path, 1)
        (seg,) = row["segments"]
        self.assertEqual((seg["prompt_ms"], seg["decode_ms"], seg["drafts_offered"], seg["drafts_accepted"]),
                         (40.0, 30.0, 5, 3))
        self.assertEqual((row["cached_tokens"], row["drafts_offered"], row["drafts_accepted"]), (7, 5, 3))
        self.assertEqual(row["expert_hit_rate"], 0.9)
        self.assertEqual(row["decode_tok_s"], round(4 / 0.03, 1))

    def test_detail_has_the_prompt_reads_progress(self):
        tok = ByteTokenizer()

        class Chunked(ClockedEngine):
            """Reads the prompt in two PP chunks (one heartbeat each), as StrataEngine's solo path yields them."""

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                for done, ms in ((len(ids) // 2, 12.5), (len(ids), 25.0)):
                    self.progress, self.progress_ms = (done, len(ids)), ms
                    yield None
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        engine = Chunked(tok, "</think>\n\nok", max_context=CTX)
        engine.progress = (1, 1)                                          # the previous request's: not this one's
        srv = self.server(engine, detail=True)
        self.assertEqual(srv.post("/v1/chat/completions", chat())[0], 200)
        (row,) = lines(self.path, 1)
        n = row["prompt_tokens"]
        self.assertEqual(row["prefill_progress"], [[n // 2, n, 12.5], [n, n, 25.0]])
        self.assertNotIn(SECRET_PROMPT, self.path.read_text(encoding="utf-8"))

    def test_with_the_monitor_on_both_work(self):
        tok = ByteTokenizer()
        srv = self.server(ClockedEngine(tok, "</think>\n\n" + SECRET_ANSWER, max_context=CTX), monitor=True)
        self.assertEqual(srv.post("/v1/chat/completions", chat())[0], 200)
        (row,) = lines(self.path, 1)
        self.assertEqual(row["status"], 200)
        record = srv.svc.request_records(row["id"])                    # the monitor's record of the same request
        self.assertIn(SECRET_PROMPT, record["input"])
        self.assertIn(SECRET_ANSWER, record["output"])
        self.assertNotIn("_log", record)
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn(SECRET_PROMPT, raw)
        self.assertNotIn(SECRET_ANSWER, raw)

    def test_log_alone_keeps_no_text_in_memory(self):
        tok = ByteTokenizer()
        srv = self.server(ClockedEngine(tok, "</think>\n\nok", max_context=CTX))
        record = srv.svc.begin_request("/v1/chat/completions", chat(), "127.0.0.1", "ua")
        srv.svc.request_trace.record = None
        self.assertEqual(len(srv.svc.api_requests), 0)
        self.assertFalse({"input", "output", "reasoning", "response"} & set(record))
        self.assertNotIn(SECRET_PROMPT, json.dumps({k: v for k, v in record.items() if k != "_clock"}))


class _Reached(Exception):
    """raised by the patched serve(): main() got as far as serving."""


class Startup(unittest.TestCase):
    """The switches through main(): the flag, $STRATA_REQUEST_LOG and the config's keys; off without them."""

    def run_main(self, cfg, args=(), env=None):
        import serve.server as S
        seen = {}

        def fake_serve(svc, host=None, port=None):
            seen["log"] = svc.request_log
            raise _Reached

        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env or {}), \
                mock.patch.object(S, "serve", fake_serve), contextlib.redirect_stdout(io.StringIO()) as out:
            for k in ("STRATA_REQUEST_LOG", "STRATA_REQUEST_LOG_DETAIL", "STRATA_API_KEY"):
                if k not in (env or {}):
                    os.environ.pop(k, None)
            cfg = {k: (v.replace("{d}", d) if isinstance(v, str) else v) for k, v in cfg.items()}
            p = Path(d) / "cfg.json"
            p.write_text(json.dumps(cfg), encoding="utf-8")
            argv = ["server.py", "--engine", "mock", "--port", "0", "--config", str(p)] + \
                [a.replace("{d}", d) for a in args]
            for k, v in list(os.environ.items()):
                if "{d}" in v:
                    os.environ[k] = v.replace("{d}", d)
            with mock.patch.object(sys, "argv", argv):
                try:
                    S.main()
                except _Reached:
                    pass
            log = seen.get("log")
            got = None if log is None else (Path(log.path).relative_to(d).as_posix(), log.max_bytes, log.backups,
                                            log.detail)
            if log is not None:
                log.close()
        return got, out.getvalue()

    def test_off_by_default(self):
        got, out = self.run_main({})
        self.assertIsNone(got)
        self.assertNotIn("request log", out)

    def test_the_flag_the_variable_and_the_config(self):
        mb = 64 * 2**20
        self.assertEqual(self.run_main({}, ["--request-log", "{d}/a.jsonl"])[0], ("a.jsonl", mb, 3, False))
        self.assertEqual(self.run_main({}, env={"STRATA_REQUEST_LOG": "{d}/b.jsonl",
                                                "STRATA_REQUEST_LOG_DETAIL": "1"})[0], ("b.jsonl", mb, 3, True))
        got, out = self.run_main({"request_log": "{d}/c.jsonl", "request_log_max_mb": 0, "request_log_backups": 5,
                                  "request_log_detail": True})
        self.assertEqual(got, ("c.jsonl", 0, 5, True))
        self.assertIn("request log on", out)
        self.assertIn("not rotated", out)
        got = self.run_main({"request_log": "{d}/c.jsonl", "request_log_max_mb": 1},
                            ["--request-log", "{d}/d.jsonl", "--request-log-max-mb", "2", "--request-log-backups", "1",
                             "--request-log-detail"])[0]
        self.assertEqual(got, ("d.jsonl", 2 * 2**20, 1, True))             # the flags win over the config

    def test_a_bad_value_stops_the_start(self):
        with self.assertRaises(SystemExit) as e:
            self.run_main({"request_log": "{d}/c.jsonl", "request_log_max_mb": "big"})
        self.assertIn("request_log_max_mb", str(e.exception))


class Rotation(unittest.TestCase):
    def test_rotates_by_size_and_keeps_backups(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.jsonl")
            log = reqlog.RequestLog(path, max_mb=1000 / 2**20, backups=2)      # about 1000 bytes a file
            try:
                for i in range(60):
                    log.write({"n": i, "pad": "x" * 60})
            finally:
                log.close()
            self.assertEqual(sorted(os.listdir(d)), ["r.jsonl", "r.jsonl.1", "r.jsonl.2"])
            seen = []
            for name in ("r.jsonl.2", "r.jsonl.1", "r.jsonl"):
                text = Path(d, name).read_text(encoding="utf-8")
                self.assertLessEqual(len(text.encode()), 1000)
                seen += [json.loads(x)["n"] for x in text.splitlines()]    # whole lines only
            self.assertEqual(seen, list(range(60 - len(seen), 60)))          # the newest, in order

    def test_no_rotation_and_copytruncate(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.jsonl")
            log = reqlog.RequestLog(path, max_mb=0)
            try:
                for i in range(50):
                    log.write({"n": i, "pad": "x" * 60})
                self.assertEqual(os.listdir(d), ["r.jsonl"])
                with open(path, "r+") as f:                                 # what logrotate's copytruncate does
                    f.truncate(0)
                log.write({"n": "after"})
            finally:
                log.close()
            self.assertEqual(Path(path).read_text(encoding="utf-8"), '{"n":"after"}\n')

    def test_a_failed_write_is_said_once_and_never_raised(self):
        with tempfile.TemporaryDirectory() as d:
            log = reqlog.RequestLog(os.path.join(d, "r.jsonl"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(log, "_open", side_effect=OSError("disk full")):
                log.f.close()                                               # every write now fails
                log.write({"n": 1})
                log.write({"n": 2})
            self.assertEqual(out.getvalue().count("request log could not be written"), 1)


if __name__ == "__main__":
    unittest.main()
