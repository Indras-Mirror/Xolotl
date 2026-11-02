"""local (Indra's Mirror): the voice sidecar - ASR (and later TTS) as a server Strata owns.

Voice duplex needs speech models the LLM engine does not carry: an ASR (audio -> text) and, for spoken
answers, a TTS.  They are small (a 0.6-0.8B decoder plus a ~0.2B audio encoder) but they are separate
weights, so running them as always-on servers means two more processes fighting the LLM for VRAM.

Instead this mirrors the lazy image encoder (`Vision` + `Service.start_lazy_vision`): Strata starts the
audio server on the first request that needs it and stops it again after `lazy_idle_s`, so

  * one place (`Service`) schedules the GPU: before the child starts the elastic expert cache is shrunk to
    leave room, and after it stops the cache grows back to the load reserve;
  * the audio model is HOT-SWAPPABLE with the LLM - its GGUF stays in the OS page cache between stops, so
    the next start is fast;
  * the child is MODEL-AGNOSTIC: it is any OpenAI-compatible audio server (llama-server with an audio
    mmproj for ASR, an audio.cpp/omnivoice server for TTS), named only by `cmd`/`args` in the config.

This module is deliberately self-contained (no `serve.server` import, which would be a cycle): Strata
imports it, not the other way round.
"""
from __future__ import annotations

import subprocess
import threading
import time
import urllib.error
import urllib.request

# A short probe: a healthy local audio server answers /health at once; if it does not, it is not up yet.
HEALTH_TIMEOUT_S = 2.0
# How long a single forwarded request may take (a long clip on a busy card is still bounded).
FORWARD_TIMEOUT_S = 120.0


class VoiceServer:
    """One lazily-started audio server (ASR or TTS).  `ensure()` brings it up; `stop()` takes it down."""

    def __init__(self, kind: str, cfg: dict, log=None):
        self.kind = kind                                    # "asr" | "tts", for messages only
        self.cmd = cfg["cmd"]
        self.args = [str(a) for a in cfg.get("args", [])]
        self.port = int(cfg.get("port", 5110 if kind == "asr" else 5210))
        self.health_path = str(cfg.get("health", "/health"))
        # the child's own route for the forwarded path, when it differs (llama-server speaks at /tts, not
        # /v1/audio/speech; the OpenAI body's input / voice / response_format are the same keys)
        self.path = str(cfg["path"]) if cfg.get("path") else None
        self.start_timeout_s = float(cfg.get("start_timeout_s", 120))
        self.reserve_mib = int(cfg.get("lazy_reserve_mib", 1800))   # VRAM the child needs, freed before it starts
        self.proc = None
        self.stopped = True
        self.lock = threading.Lock()
        self.last_used = 0.0
        self.on_start = None                               # set by Service.start_lazy_voice (shrinks the cache)
        self._log = log

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.proc is not None and not self.stopped and self.proc.poll() is None

    def ensure(self) -> str:
        """Start the child if it is down and wait for its health check.  Returns the base URL.
        Raises RuntimeError if it cannot be made healthy.  The lock is held across the start so two
        requests cannot both spawn it (a second request is rare - Strata serves one turn at a time)."""
        with self.lock:
            if self.alive():
                return self.base_url
            if self.on_start is not None:
                try:
                    self.on_start()
                except Exception as e:                      # the cache shrink is best-effort, never fatal
                    self._say(f"could not make room for the {self.kind} server ({e}); starting it anyway")
            args = [self.cmd, *[a.format(port=self.port) for a in self.args]]
            self._say(f"starting the {self.kind} server on port {self.port} (lazy)")
            try:
                self.proc = subprocess.Popen(               # noqa: S603 - the command comes from the config
                    args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=self._log or subprocess.DEVNULL, text=True)
            except OSError as e:
                self.proc = None
                raise RuntimeError(f"the {self.kind} server could not start: {e}") from None
            self.stopped = False
            # the start counts as use: until the caller's touch() after ensure(), last_used was 0 (or the last
            # idle stop), so an idle-loop tick landing during a slow cold start stopped the child under the
            # request that started it (seen: HTTP 000 on the first transcription; the audio path holds no FIFO)
            self.last_used = time.time()
            try:
                self._wait_healthy()
            except RuntimeError:
                self.stop()                                 # do not leave a half-started child behind
                raise
            self.last_used = time.time()
            self._say(f"the {self.kind} server is up on port {self.port}")
            return self.base_url

    def _wait_healthy(self) -> None:
        deadline = time.time() + self.start_timeout_s
        while time.time() < deadline:
            if not self.alive():
                raise RuntimeError(f"the {self.kind} server exited during startup")
            if self._healthy():
                return
            time.sleep(0.5)
        raise RuntimeError(f"the {self.kind} server did not answer {self.health_path} in {self.start_timeout_s:.0f} s")

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + self.health_path, timeout=HEALTH_TIMEOUT_S) as r:  # noqa: S310
                return r.status == 200
        except Exception:
            return False

    def touch(self) -> None:
        self.last_used = time.time()

    def stop(self) -> None:
        """Stop the child (its VRAM goes back; its GGUF stays in the page cache for the next start)."""
        with self.lock:
            proc, self.proc, self.stopped = self.proc, None, True
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def forward(self, path: str, body: bytes, content_type: str, timeout: float = FORWARD_TIMEOUT_S):
        """POST `body` to the child at `path`; return (status, bytes, content_type).  Never raises on an
        HTTP error status - the child's own error body is passed back to the caller."""
        req = urllib.request.Request(self.base_url + (self.path or path), data=body, method="POST")  # noqa: S310
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
                return r.status, r.read(), r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("Content-Type", "application/json")

    def _say(self, message: str) -> None:
        print(f"[strata] {message}", flush=True)


def start_lazy_voice(service, servers: dict, idle_s: float) -> None:
    """Wire the lazy start/stop for every voice server in `servers` (mirrors start_lazy_vision).

    `service` is the Strata Service: it supplies `.fifo` (so a child is stopped between turns, never
    mid-request), `.loaded()`, and the elastic-cache calls `.engine.vram(mib)` / `.vram_reserve`."""
    if not servers:
        return
    elastic = hasattr(service.engine, "vram") and \
        bool((getattr(service.engine, "info", {}) or {}).get("vram_elastic", True))

    for name, v in servers.items():
        reserve = v.reserve_mib

        def make_room(reserve=reserve, name=name):
            if not elastic or not service.loaded():
                return
            try:
                out = service.engine.vram(reserve)
                print(f"[strata] lazy voice ({name}): expert cache shrunk ({out.get('expert_slots')} slots, "
                      f"{out.get('vram_free_mib')} MiB free)", flush=True)
            except Exception as e:
                print(f"[strata] lazy voice ({name}): could not shrink the cache ({e}); starting anyway", flush=True)
        v.on_start = make_room

    wanted = ", ".join(sorted(servers))
    print(f"[strata] lazy voice ({wanted}): started with the first request, stopped after {idle_s:.0f} s idle",
          flush=True)

    def loop():
        while True:
            time.sleep(max(1.0, min(15.0, idle_s / 4)))
            for name, v in servers.items():
                try:
                    if not v.alive() or time.time() - v.last_used < idle_s:
                        continue
                    if not service.fifo.acquire(timeout=1.0):   # a request is running: the next turn
                        continue
                    try:
                        if not v.alive() or time.time() - v.last_used < idle_s:
                            continue
                        v.stop()
                        elastic = hasattr(service.engine, "vram") and \
                            bool((getattr(service.engine, "info", {}) or {}).get("vram_elastic", True))
                        out = service.engine.vram(service.vram_reserve) if elastic and service.loaded() else {}
                        print(f"[strata] lazy voice: {name} server stopped after {idle_s:.0f} s idle; "
                              f"expert cache grown back ({out.get('expert_slots')} slots)", flush=True)
                    finally:
                        service.fifo.release()
                except Exception as e:                          # the thread keeps running
                    print(f"[strata] lazy voice ({name}): {e}", flush=True)

    threading.Thread(target=loop, daemon=True, name="strata-lazy-voice").start()
