"""Bounded JSON transport to the optional, ABI-isolated PR #31192 mtmd helper.

The helper owns its llama.cpp/ggml libraries and never runs the LLM. Every
transport/startup failure becomes a request error; it cannot strand the text
engine waiting for a second process. Linux is the validated pipe platform.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import threading
import time
import uuid


class VisionUnavailable(RuntimeError):
    """An optional encoder failure, safe to report without stopping text."""


class VisionEncoder:
    MAX_REPLY = 4 * 1024 * 1024
    MAX_REQUEST = 16 * 1024 * 1024

    def __init__(self, adapter, model, mmproj, cache_dir, *, backend="cpu",
                 threads=16, timeout=180, startup_timeout=180,
                 embedding_cache_mb=256, env=None):
        if backend not in ("cpu", "cuda"):
            raise ValueError("vision backend must be cpu or cuda")
        for name, value in (("timeout", timeout), ("startup_timeout", startup_timeout)):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"vision {name} must be finite and positive")
        if type(threads) is not int or not 1 <= threads <= 128:
            raise ValueError("vision threads must be an integer in 1..128")
        if type(embedding_cache_mb) is not int or not 0 <= embedding_cache_mb <= 4096:
            raise ValueError("vision embedding_cache_mb must be an integer in 0..4096")
        self.adapter = Path(adapter).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.timeout, self.startup_timeout = float(timeout), float(startup_timeout)
        self.env = dict(os.environ if env is None else env)
        self.config = {"model": str(Path(model).expanduser().resolve()),
                       "mmproj": str(Path(mmproj).expanduser().resolve()),
                       "backend": backend, "threads": threads, "encode": True,
                       "encoder_mode": "single", "add_special": False,
                       "embedding_cache_mb": embedding_cache_mb}
        self._lock = threading.RLock()
        self.process = self.log = None
        self._config_path = None
        self._buffer = bytearray()
        self.last_stats = {}
        self.startup_stats = {}

    @staticmethod
    def _error(message):
        return VisionUnavailable("vision unavailable: " + str(message))

    def _wait_pipe(self, file, event, deadline, cancel=None):
        with selectors.DefaultSelector() as selector:
            selector.register(file, event)
            while True:
                if cancel is not None and cancel.is_set():
                    raise self._error("helper request cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._error("helper timed out")
                if selector.select(min(remaining, 0.1) if cancel is not None else remaining):
                    return

    def _read(self, deadline, cancel=None):
        # select()+readline() is insufficient: an alive worker may emit one
        # byte without a newline and then hang. Read nonblocking chunks under
        # one absolute deadline, including bytes already buffered by the OS.
        while b"\n" not in self._buffer:
            if len(self._buffer) >= self.MAX_REPLY:
                raise self._error("helper reply exceeds the size limit")
            self._wait_pipe(self.process.stdout, selectors.EVENT_READ, deadline, cancel)
            try:
                block = os.read(self.process.stdout.fileno(), min(65536, self.MAX_REPLY-len(self._buffer)))
            except BlockingIOError:
                continue
            if not block:
                raise self._error(f"helper exited ({self.process.poll()}); see encoder.log")
            self._buffer.extend(block)
        line, _, rest = self._buffer.partition(b"\n")
        self._buffer = bytearray(rest)
        try:
            reply = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise self._error("invalid helper JSON reply") from exc
        if not isinstance(reply, dict):
            raise self._error("helper reply is not an object")
        return reply

    def _write(self, request, deadline, cancel=None):
        data = memoryview((json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        if len(data) > self.MAX_REQUEST:
            raise self._error("helper request exceeds the size limit")
        while data:
            self._wait_pipe(self.process.stdin, selectors.EVENT_WRITE, deadline, cancel)
            try:
                count = os.write(self.process.stdin.fileno(), data)
            except BlockingIOError:
                continue
            if count <= 0:
                raise self._error("helper input pipe closed")
            data = data[count:]

    def _start(self, cancel=None):
        if cancel is not None and cancel.is_set():
            raise self._error("helper request cancelled")
        if self.process is not None and self.process.poll() is None:
            return False
        self._close()
        if not self.adapter.is_file() or not os.access(self.adapter, os.X_OK):
            raise self._error(f"helper executable is missing or not executable: {self.adapter}")
        for key in ("model", "mmproj"):
            if not Path(self.config[key]).is_file():
                raise self._error(f"{key} file is missing: {self.config[key]}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        config = self.cache_dir / ("worker-" + uuid.uuid4().hex + ".json")
        self._config_path = config
        config.write_text(json.dumps(self.config) + "\n", encoding="utf-8")
        self.log = (self.cache_dir / "encoder.log").open("ab", buffering=0)
        env = dict(self.env)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = str(self.config["threads"])
        if self.config["backend"] == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        # No shell, and no PR libraries are loaded into this Python process.
        self.process = subprocess.Popen([str(self.adapter), "--worker", str(config)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, bufsize=0, env=env,
                                        start_new_session=os.name == "posix")
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        reply = self._read(time.monotonic() + self.startup_timeout, cancel)
        if reply.get("ready") is not True:
            raise self._error(reply.get("error", "helper failed to initialize"))
        self.startup_stats = reply.get("startup_stages_ms", {})
        return True

    def start(self, cancel=None):
        with self._lock:
            started = time.monotonic()
            try:
                fresh = self._start(cancel)
                return {"worker_started": fresh, "milliseconds": (time.monotonic()-started)*1000,
                        "backend": self.config["backend"], "startup_stages_ms": self.startup_stats}
            except Exception as exc:
                self._close()
                if isinstance(exc, VisionUnavailable):
                    raise
                raise self._error(exc) from exc

    def encode(self, prompt, media, *, video_fps=1.0, timestamp_interval_ms=5000, cancel=None):
        with self._lock:
            started = time.monotonic()
            out = None
            try:
                if not isinstance(prompt, str) or not media:
                    raise ValueError("a prompt and at least one media path are required")
                if isinstance(video_fps, bool) or not math.isfinite(float(video_fps)) or not 0 < video_fps <= 30:
                    raise ValueError("video_fps must be in (0, 30]")
                if type(timestamp_interval_ms) is not int or not 1 <= timestamp_interval_ms <= 2**31-1:
                    raise ValueError("timestamp_interval_ms must be a positive int32")
                paths = [str(Path(path).resolve(strict=True)) for path in media]
                fresh = self._start(cancel)
                out = self.cache_dir / ("request-" + uuid.uuid4().hex)
                request = {"prompt": prompt, "media": paths, "out": str(out), "encode": True,
                           "video_fps": video_fps, "timestamp_interval_ms": timestamp_interval_ms}
                # One deadline covers backpressure on stdin AND the reply.
                deadline = time.monotonic() + self.timeout
                self._write(request, deadline, cancel)
                reply = self._read(deadline, cancel)
                if reply.get("ok") is not True:
                    raise self._error(reply.get("error", "helper encoding failed"))
                manifest = Path(reply["manifest"]).resolve(strict=True)
                if manifest != out / "manifest.json" or not manifest.is_file():
                    raise self._error("helper returned an unexpected manifest path")
                self.last_stats = {"worker_started": fresh, "seconds": time.monotonic()-started,
                                   "backend": self.config["backend"], "rows": reply.get("rows"),
                                   "visual_rows": reply.get("visual_rows"),
                                   "timings_ms": reply.get("timings_ms", {})}
                return manifest
            except Exception as exc:
                # Never reuse a worker whose protocol state may be partial.
                self._close()
                if out is not None:
                    # This UUID path belongs only to this failed request.
                    # Successful manifests are released by the consuming shim.
                    shutil.rmtree(out, ignore_errors=True)
                if isinstance(exc, VisionUnavailable):
                    raise
                raise self._error(exc) from exc

    prepare = encode

    def _close(self):
        process, self.process = self.process, None
        self._buffer.clear()
        if process is not None:
            try:
                if process.poll() is None:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait(timeout=1)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            finally:
                for pipe in (process.stdin, process.stdout):
                    if pipe:
                        pipe.close()
        if self.log is not None:
            self.log.close()
            self.log = None
        if self._config_path is not None:
            try:
                self._config_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._config_path = None

    def close(self):
        with self._lock:
            self._close()

    unload = close

    def restart(self):
        with self._lock:
            self._close()
            return self.start()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
