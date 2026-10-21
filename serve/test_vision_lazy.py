"""The lazy image encoder ("lazy": true in the vision section) on a card the engine has filled: an encoder that cannot
start is a 503 that names the cause (here a GPU out of memory), not a dropped connection; the cause is read from this
start's own lines of a shared log; a lazy encoder on the GPU without "vram_elastic" starts with the model instead; and
the cache is asked to make room only when the engine can shrink it.

    python -m serve.test_vision_lazy
"""
import base64
import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, Vision, serve
from serve.test_vram import VramEngine

ROOT = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 32          # the header is all Vision.normalize looks at
OOM = ("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 865.48 MiB on device 0: cudaMalloc failed: "
       "out of memory\n")


def encoder(log_path, stdout, stderr=""):
    """popen's stand-in: the encoder writes `stderr` to the shared log and `stdout` to its pipe, then ends."""
    def start(*a, **k):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(stderr)
        return SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(stdout), kill=lambda: None,
                               wait=lambda timeout=None: 0, poll=lambda: 1)
    return start


class LazyStart(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.log_path = Path(d) / "engine.log"
        self.log_path.write_text("strata serve: an earlier line of the engine\n", encoding="utf-8")
        self.log = open(self.log_path, "a", encoding="utf-8")
        self.addCleanup(self.log.close)
        self.vision = Vision({"exe": "strata-vision", "mmproj": "m.gguf", "model": "t.gguf", "gpu": True,
                              "lazy": True}, log=self.log)

    def start(self, stdout, stderr=""):
        with mock.patch.object(server, "popen", encoder(self.log_path, stdout, stderr)), \
                mock.patch.object(server, "contain"):
            self.vision.encode(PNG)

    def test_out_of_memory_is_named(self):
        # what the encoder did on a card the expert cache had filled: ggml's abort, then gdb's line on stdout
        with self.assertRaises(server.VisionUnavailable) as cm:
            self.start("[New LWP 3809159]\n", "load_hparams: ...\n" + OOM +
                       "alloc_tensor_range: failed to allocate CUDA0 buffer of size 907524736\n")
        m = str(cm.exception)
        self.assertIn("ran out of memory", m)
        self.assertIn("cudaMalloc failed: out of memory", m)
        self.assertIn('"vram_elastic": true', m)
        self.assertIn("[New LWP 3809159]", m)                  # the encoder's own first line is still quoted
        self.assertFalse(self.vision.alive())                   # and the next picture starts it again

    def test_older_lines_of_the_shared_log_are_not_the_cause(self):
        with open(self.log_path, "a", encoding="utf-8") as f:     # an earlier start's lines
            f.write(OOM + "cudaMalloc failed: CUDA-capable device(s) is/are busy or unavailable\n")
        with self.assertRaises(server.VisionUnavailable) as cm:
            self.start("ERR cannot open m.gguf\n")
        self.assertEqual(str(cm.exception), "the vision encoder did not start: ERR cannot open m.gguf")

    def test_an_encoder_that_starts_is_used(self):
        lines = ["READY\n", "OK 3\n"]
        out = SimpleNamespace(readline=lambda: lines.pop(0) if lines else "")

        def start(*a, **k):
            return SimpleNamespace(stdin=io.StringIO(), stdout=out, kill=lambda: None, wait=lambda timeout=None: 0,
                                   poll=lambda: None)
        with mock.patch.object(server, "popen", start), mock.patch.object(server, "contain"):
            path, n = self.vision.encode(PNG)
        self.assertEqual(n, 3)
        self.assertTrue(self.vision.alive())


class Http(unittest.TestCase):
    """The request with the picture gets an answer: a 503 with the message, on /v1/chat/completions and /v1/messages."""

    def setUp(self):
        d = tempfile.mkdtemp()
        self.log_path = Path(d) / "engine.log"
        self.log = open(self.log_path, "a", encoding="utf-8")
        self.addCleanup(self.log.close)
        tok = ByteTokenizer()
        vision = Vision({"exe": "strata-vision", "mmproj": "m.gguf", "model": "t.gguf", "gpu": True, "lazy": True},
                        log=self.log)
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=vision)
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, path, body):
        r = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_a_503_with_the_cause(self):
        url = "data:image/png;base64," + base64.b64encode(PNG).decode()
        openai = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                            {"type": "text", "text": "read it"}]}], "max_tokens": 8}
        anthropic = {"max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(PNG).decode()}},
            {"type": "text", "text": "read it"}]}]}
        for path, body in (("/v1/chat/completions", openai), ("/v1/chat/completions", {**openai, "stream": True}),
                           ("/v1/messages", anthropic)):
            with self.subTest(path=path, stream=body.get("stream", False)):
                with mock.patch.object(server, "popen", encoder(self.log_path, "[New LWP 1]\n", OOM)), \
                        mock.patch.object(server, "contain"), contextlib.redirect_stdout(io.StringIO()):
                    status, out = self.post(path, body)
                self.assertEqual(status, 503)
                self.assertIn("ran out of memory", json.dumps(out))
        # and text keeps working
        status, out = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                         "max_tokens": 8})
        self.assertEqual(status, 200)


class Config(unittest.TestCase):
    def quiet(self, cfg, vcfg):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            return server.on_demand_vision(cfg, vcfg), out.getvalue()

    def test_lazy_on_the_gpu_needs_an_elastic_cache(self):
        v, said = self.quiet({"args": []}, {"gpu": True, "lazy": True})
        self.assertFalse(v["lazy"])
        self.assertIn('"vram_elastic": true', said)
        for cfg in ({"vram_elastic": True, "args": []}, {"args": ["--vram-elastic"]}):
            v, said = self.quiet(cfg, {"gpu": True, "lazy": True})
            self.assertTrue(v["lazy"])
            self.assertEqual(said, "")

    def test_lazy_on_the_cpu_needs_nothing(self):
        v, said = self.quiet({"args": []}, {"lazy": True})
        self.assertTrue(v["lazy"])
        self.assertEqual(said, "")

    def test_elastic_is_unchanged(self):
        v, said = self.quiet({"args": []}, {"elastic": True})
        self.assertFalse(v["elastic"])
        self.assertIn('"elastic" needs', said)
        v, _ = self.quiet({"vram_elastic": True}, {"elastic": True, "gpu": True})
        self.assertTrue(v["elastic"])


class MakeRoom(unittest.TestCase):
    def service(self, vram_elastic):
        tok = ByteTokenizer()
        engine = VramEngine(tok)
        engine.loaded = True
        engine.info["vram_elastic"] = vram_elastic
        vision = SimpleNamespace(lazy_vision=True, on_start=None, alive=lambda: False, last_used=0.0)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=vision)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(server.threading, "Thread"):
            svc.start_lazy_vision(2400, 300)
        return svc, engine, vision

    def test_an_elastic_engine_shrinks_its_cache_first(self):
        svc, engine, vision = self.service(1)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            vision.on_start()
        self.assertEqual(engine.calls, [2400])
        self.assertIn("expert cache shrunk", out.getvalue())

    def test_an_engine_that_cannot_shrink_says_so(self):
        svc, engine, vision = self.service(0)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            vision.on_start()
        self.assertEqual(engine.calls, [])
        self.assertIn("cannot shrink", out.getvalue())


if __name__ == "__main__":
    unittest.main()
