#!/usr/bin/env python3
"""One-shot capture of what QuetzaCodetl actually sends to a local model (system prompt, tool schemas, messages).

usage: harness_capture_server.py <out-dir> [port=8199]

Answers /health, /v1/models and /v1/messages (Anthropic Messages API, what Quetza uses against Xolotl serve) with a
minimal valid reply ("ok", end_turn), and writes every request body to <out-dir>/req-<n>.json. Point Quetza at it with
the same env xol-qfn uses (ANTHROPIC_BASE_URL / QUETZA_ENDPOINT = http://127.0.0.1:<port>) and run one print-mode
prompt. Nothing leaves the machine; no model runs.
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OUT = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8199
N = [0]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"data": [{"id": "Swift15-QFN-XQ-45GB-Xolotl", "object": "model",
                                  "meta": {"n_ctx": 524288}}]})
        else:
            self._json({"status": "ok"})

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n)
        N[0] += 1
        path = f"{OUT}/req-{N[0]:03d}.json"
        with open(path, "wb") as f:
            f.write(body)
        print(f"captured {self.path} -> {path} ({n} bytes)", flush=True)
        try:
            req = json.loads(body)
        except Exception:
            req = {}
        if req.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            mid = f"msg_{int(time.time())}"
            ev = [
                ("message_start", {"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant",
                  "model": req.get("model", "x"), "content": [], "stop_reason": None, "stop_sequence": None,
                  "usage": {"input_tokens": 1, "output_tokens": 0}}}),
                ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                  "usage": {"output_tokens": 1}}),
                ("message_stop", {"type": "message_stop"}),
            ]
            for name, data in ev:
                self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
        else:
            self._json({"id": "msg_x", "type": "message", "role": "assistant", "model": req.get("model", "x"),
                        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1}})


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
