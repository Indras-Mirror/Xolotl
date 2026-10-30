"""serve/request_log.py - the opt-in request log: one JSON line per generation request, for statistics.

Off unless `--request-log PATH` (or $STRATA_REQUEST_LOG, or "request_log" in the config) names a file.  Each finished
request to /v1/chat/completions, /v1/messages or /v1/responses appends one line: when it came, how long it waited for
its turn, its time to first token, the prompt tokens the conversation cache held and the ones read, the tokens written,
the prompt and decode speeds, how it ended, and its pictures.  No request or answer text, no headers but the
User-Agent, and no API key are written.

The file is opened for appending and every line is flushed whole, so `logrotate` with `copytruncate` is safe.  Built-in
rotation (on by default, `--request-log-max-mb`, 0 turns it off) renames the file to PATH.1 (PATH.1 to PATH.2, ...)
once the next line would take it over the size; `--request-log-backups` keeps that many.  A write that fails is said
once in the server window and never fails a request.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone

DEFAULT_MAX_MB = 64
DEFAULT_BACKUPS = 3
USER_AGENT_MAX = 200
ERROR_MAX = 300
PREFILL_SAMPLES_MAX = 64          # detail: the prompt read's progress, at most this many chunks per request

API_OF = {"/v1/chat/completions": "openai", "/v1/messages": "anthropic", "/v1/responses": "responses"}


def utc_iso(t: float) -> str:
    """Seconds since the epoch as UTC ISO 8601 with milliseconds: 2026-10-29T11:56:22.123Z."""
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class RequestLog:
    """Appends JSON lines to `path`, rotating by size.  Thread-safe: one lock around each line."""

    def __init__(self, path: str, max_mb: float = DEFAULT_MAX_MB, backups: int = DEFAULT_BACKUPS,
                 detail: bool = False):
        if not isinstance(path, str) or not path.strip() or any(ord(c) < 32 for c in path):
            raise ValueError("request_log must be a file path")
        if isinstance(max_mb, bool) or not isinstance(max_mb, (int, float)) or max_mb < 0:
            raise ValueError("request_log_max_mb must be a number of MB (0 or more; 0 = no rotation)")
        if isinstance(backups, bool) or not isinstance(backups, int) or backups < 1:
            raise ValueError("request_log_backups must be a whole number (1 or more)")
        self.path = os.path.abspath(path)
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.backups = backups
        self.detail = bool(detail)
        self.lock = threading.Lock()
        self.failed = False
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.f = self._open()                             # OSError here: the start says so and stops

    def _open(self):
        return open(self.path, "a", encoding="utf-8", newline="\n")

    def _rotate(self) -> None:
        self.f.close()
        for i in range(self.backups - 1, 0, -1):
            src = f"{self.path}.{i}"
            if os.path.exists(src):
                os.replace(src, f"{self.path}.{i + 1}")
        os.replace(self.path, self.path + ".1")
        self.f = self._open()

    def write(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.lock:
            try:
                if self.max_bytes:
                    size = self.f.tell()
                    if size and size + len(line.encode("utf-8")) > self.max_bytes:
                        self._rotate()
                self.f.write(line)
                self.f.flush()
            except (OSError, ValueError) as e:            # a full disk, a removed folder: said once, never raised
                if not self.failed:
                    self.failed = True
                    print(f"[strata] the request log could not be written ({e}); requests are not affected",
                          flush=True)
                try:
                    self.f.close()
                    self.f = self._open()
                except OSError:
                    pass

    def close(self) -> None:
        with self.lock:
            try:
                self.f.close()
            except OSError:
                pass


def new_stats(queued: int, client: str | None, user_agent: str | None, detail: bool) -> dict:
    """The per-request figures the server fills in as the request goes (Service._log_run and friends)."""
    s = {"queued_at_arrival": queued, "client": client,
         "user_agent": (user_agent or "")[:USER_AGENT_MAX] or None, "runs": 0}
    if detail:
        s["segments"], s["prefill_progress"] = [], []
    return s


def add_run(stats: dict, *, prompt_tokens: int, max_tokens, thinking: bool, generated: int, finish: str,
            last: dict, timings: dict | None, segments: list, start_s: float) -> None:
    """One engine generation of this request (MCP's tool rounds run several) into its figures."""
    stats["runs"] += 1
    stats.setdefault("start_s", start_s)
    stats["max_tokens"] = max_tokens
    stats["thinking"] = thinking
    stats["finish"] = finish
    for key, value in (("prompt_tokens", prompt_tokens), ("output_tokens", generated)):
        stats[key] = stats.get(key, 0) + value
    reused = last.get("reused")
    if reused is not None:
        stats["cached_tokens"] = stats.get("cached_tokens", 0) + int(reused)
        stats.setdefault("first_reused", int(reused))
    if timings is not None:
        stats["read_tokens"] = stats.get("read_tokens", 0) + int(timings.get("prompt_n") or 0)
    for key in ("prompt_ms", "decode_ms", "drafts_offered", "drafts_accepted"):
        if last.get(key) is not None:
            stats[key] = stats.get(key, 0) + last[key]
    if last.get("generated") is not None:
        stats["engine_generated"] = stats.get("engine_generated", 0) + int(last["generated"])
    if "segments" in stats:
        stats["segments"].extend(dict(x) for x in segments)


def entry(record: dict, stats: dict, detail: bool = False) -> dict:
    """The JSON line for one finished request: `record` is the monitor's trace record (serve/server.py's
    begin_request), `stats` the figures new_stats/add_run collected.  Fields that were not measured are left out."""
    start = record["started_at"]
    total = record.get("wallclock_s")
    out = {"time": utc_iso(start), "id": record["id"], "endpoint": record["path"],
           "api": API_OF.get(record["path"]), "model": record.get("model"), "stream": record.get("stream", False),
           "client": stats.get("client"), "user_agent": stats.get("user_agent"),
           "status": record.get("http_status"), "outcome": record.get("state"),
           "queued_at_arrival": stats.get("queued_at_arrival")}
    err = record.get("error")
    if isinstance(err, dict):
        out["error_type"] = err.get("type") or err.get("code")
        out["error"] = str(err.get("message") or "")[:ERROR_MAX]
    if total is not None:
        out["end"] = utc_iso(start + total)
        out["total_s"] = total
    out["queue_s"] = record.get("queue_s")
    if record.get("load_s"):
        out["load_s"] = record["load_s"]
    for key in ("image_wait_s", "image_encode_s", "start_s"):
        if stats.get(key) is not None:
            out[key] = round(stats[key], 3)
    if record.get("first_token_s") is not None:
        out["ttft_s"] = record["first_token_s"]
    for key in ("finish", "max_tokens", "thinking", "prompt_tokens", "cached_tokens", "read_tokens",
                "output_tokens", "reasoning_tokens", "tool_calls", "stop_sequence", "drafts_offered",
                "drafts_accepted"):
        if stats.get(key) is not None:
            out[key] = stats[key]
    if stats.get("runs", 0) > 1:
        out["runs"] = stats["runs"]
    out["cancelled"] = bool(stats.get("cancelled")) or record.get("state") == "disconnected"
    if stats.get("prompt_ms") is not None:
        out["prefill_ms"] = round(stats["prompt_ms"], 1)
        if stats.get("read_tokens") and stats["prompt_ms"] > 0:
            out["prefill_tok_s"] = round(stats["read_tokens"] / (stats["prompt_ms"] / 1000), 1)
    if stats.get("decode_ms") is not None:
        out["decode_ms"] = round(stats["decode_ms"], 1)
        made = stats.get("engine_generated") or stats.get("output_tokens")
        if made and stats["decode_ms"] > 0:
            out["decode_tok_s"] = round(made / (stats["decode_ms"] / 1000), 1)
    if stats.get("images"):
        out["images"] = stats["images"]
        for key in ("images_encoded", "images_from_encoder_cache"):
            if stats.get(key) is not None:
                out[key] = stats[key]
        ends, reused = stats.get("image_ends"), stats.get("first_reused")
        if ends is not None and reused is not None:      # pictures wholly inside the prefix the cache held
            out["images_in_cached_prefix"] = sum(1 for e in ends if e <= reused)
    if detail:
        out["segments"] = stats.get("segments") or []
        if stats.get("expert_hit_rate") is not None:
            out["expert_hit_rate"] = stats["expert_hit_rate"]
        if stats.get("prefill_progress"):
            out["prefill_progress"] = stats["prefill_progress"]
    return out


def done_suffix(record: dict | None) -> str:
    """With the request log on: the time and request id after the server window's `[strata] done:` line, so the two
    can be matched (nothing otherwise, the line stays as it was)."""
    if record is None or record.get("_log") is None:
        return ""
    return f" at {utc_iso(time.time())}, request {record['id']}"
