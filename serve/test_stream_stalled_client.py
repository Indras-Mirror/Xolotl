"""serve/test_stream_stalled_client.py - #4271: a streaming client that stops reading gives the engine's turn back.

The thread that writes the SSE stream to a client also holds the engine's turn (the one-request path).  A client that
stops reading without closing - a hung process, a network that dropped - fills the socket buffers, and the write blocks
until the OS gives the connection up: the engine idle, every other request waiting.  A write that makes no progress for
STREAM_WRITE_TIMEOUT_S ends the request as a disconnect (the engine is stopped, the turn given back).

No GPU: the engine is a MockEngine that streams a lot of tokens.
"""
import json
import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class StalledClient(unittest.TestCase):
    def test_a_client_that_stops_reading_gives_the_turn_back(self):
        tok = ByteTokenizer()
        # many DISTINCT tokens: one repeated token would trip the repeat guard (#606) and end the reply
        engine = MockEngine(tok, " ".join(str(i) for i in range(80000)), max_context=2_000_000)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        stuck = socket.socket()
        try:
            with mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", 0.5, create=True):
                stuck.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                stuck.connect(("127.0.0.1", port))
                body = json.dumps({"messages": [{"role": "user", "content": "go"}], "stream": True,
                                   "max_tokens": 400000, "reasoning_effort": "none"}).encode()
                stuck.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                              b"Content-Type: application/json\r\n"
                              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                # ... and it never reads.  Wait until the server is writing into a full buffer, then until it lets go.
                end = time.monotonic() + 5
                while time.monotonic() < end and not svc.fifo.locked():
                    time.sleep(0.01)
                self.assertTrue(svc.fifo.locked(), "the stuck client's request holds the engine's turn")
                end = time.monotonic() + 20
                while time.monotonic() < end and svc.fifo.locked():
                    time.sleep(0.05)
                self.assertFalse(svc.fifo.locked(), "the turn is still held by a client that does not read")
        finally:
            stuck.close()
            httpd.shutdown()
            httpd.server_close()

    def test_the_timeout_can_be_turned_off(self):
        """STRATA_STREAM_WRITE_TIMEOUT_S=0 restores the old behaviour (no socket timeout at all).

        A fresh interpreter reads the variable, not importlib.reload(server): a reload re-executes serve/server.py in
        the same module object and rebinds every class in it, and a module that did `from serve.server import
        EngineDied` keeps the old class - its assertRaises(EngineDied) then misses the exception the reloaded code
        raises (test_vram.test_refused_and_died, test_vision_elastic's 503 test, under discover)."""
        out = subprocess.run([sys.executable, "-c", "import serve.server as s; print(s.STREAM_WRITE_TIMEOUT_S)"],
                             cwd=str(ROOT), capture_output=True, text=True, timeout=120,
                             env={**os.environ, "STRATA_STREAM_WRITE_TIMEOUT_S": "0"})
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(float(out.stdout.strip().splitlines()[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
