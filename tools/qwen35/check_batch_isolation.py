#!/usr/bin/env python3
"""Opt-in HTTP response repeatability/isolation check on already-running services.

No calls on import, --help, or without --run. This compares unconstrained
response text, finish reason and token usage; it cannot prove internal token-ID,
logit, RNG-state, GPU-overlap or numerical isolation. No lifecycle operations.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import socket
import threading
import time
import uuid
from urllib.parse import urlsplit

TOPICS = (
    "Explain how a bicycle's pedals, chain and rear wheel transfer motion. Use clear descriptive prose.",
    "Describe a quiet river passing a cedar forest after rain. Include several concrete visual details.",
    "Explain why the Moon shows phases as it orbits Earth. Distinguish illumination from Earth's shadow.",
    "Describe how a deciduous tree changes from winter to summer. Mention buds, leaves and sunlight.",
)
PROFILES = {"greedy": {"temperature": 0, "top_p": 1},
            "sampled": {"temperature": 0.9, "top_p": 0.9}}


def cases(model, profile, tokens):
    result = []
    for index, topic in enumerate(TOPICS):
        payload = {"model": model, "messages": [
            {"role": "system", "content": "Answer the requested public topic directly in natural prose. Do not output hidden reasoning."},
            {"role": "user", "content": topic}],
            **PROFILES[profile], "seed": 3100 + index, "max_tokens": tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        result.append({"case": index, "seed": payload["seed"], "payload": payload,
                       "encoded": encoded, "body_sha256": hashlib.sha256(encoded).hexdigest()})
    return result


def endpoint(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("URL must be an HTTP(S) origin or path without credentials/query/fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path += "/chat/completions"
    elif not path.endswith("/v1/chat/completions"):
        path += "/v1/chat/completions"
    return parsed, path


def parse_sse(response, row, max_tokens):
    """Strict single-choice text SSE. Malformed/error/truncated streams fail closed."""
    pending = []
    events = 0
    finish_seen = False

    def dispatch(data):
        nonlocal events, finish_seen
        events += 1
        if events > 4096:
            raise ValueError("too many SSE events")
        if data == "[DONE]":
            row["done"] = True
            return
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError("SSE event must be an object")
        row["events"].append({"monotonic_ns": time.monotonic_ns(), "event": event})
        if event.get("error"):
            raise ValueError("SSE error: " + str(event["error"]))
        if event.get("usage") is not None:
            if row["usage"] is not None and row["usage"] != event["usage"]:
                raise ValueError("conflicting usage events")
            row["usage"] = event["usage"]
        choices = event.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("expected at most one choice")
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise ValueError("invalid choice index")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise ValueError("delta must be an object")
            if any(delta.get(key) for key in ("reasoning", "reasoning_content", "tool_calls", "function_call", "refusal")):
                raise ValueError("unexpected reasoning/tool/refusal output")
            piece = delta.get("content")
            if piece is not None and not isinstance(piece, str):
                raise ValueError("content must be a string")
            if piece:
                if finish_seen:
                    raise ValueError("content after finish_reason")
                row.setdefault("first_ns", time.monotonic_ns())
                row["text"] += piece
            reason = choice.get("finish_reason")
            if reason is not None:
                if finish_seen or reason not in ("stop", "length"):
                    raise ValueError("invalid/duplicate finish_reason")
                finish_seen = True
                row["finish_reason"] = reason

    while not row["done"]:
        raw = response.readline(1048577)
        if not raw:
            break
        if len(raw) > 1048576:
            raise ValueError("oversized SSE line")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if pending:
                dispatch("\n".join(pending))
                pending.clear()
        elif line.startswith("data:"):
            pending.append(line[5:].lstrip(" "))
        elif not (line.startswith(":") or line.startswith(("event:", "id:", "retry:"))):
            raise ValueError("malformed SSE line")
    if not row["done"]:
        raise ValueError("SSE EOF without terminated [DONE] event")
    if not finish_seen or not row["text"]:
        raise ValueError("missing finish_reason or empty text")
    usage = row["usage"]
    if not isinstance(usage, dict):
        raise ValueError("missing usage")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise ValueError("invalid usage." + key)
    if usage["prompt_tokens"] < 1 or not 1 <= usage["completion_tokens"] <= max_tokens:
        raise ValueError("usage outside request bounds")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ValueError("inconsistent total_tokens")
    if row["finish_reason"] == "length" and usage["completion_tokens"] != max_tokens:
        raise ValueError("length finish below requested budget")


def request(url, case, timeout, run_deadline, request_id, barrier=None):
    row = {"request_id": request_id, "case": case["case"], "seed": case["seed"],
           "body_sha256": case["body_sha256"], "status": None, "done": False,
           "text": "", "finish_reason": None, "usage": None, "errors": [], "events": []}
    connection = None
    timer = None
    live_socket = [None]
    expired = threading.Event()

    def close():
        try:
            sock = live_socket[0] or (connection.sock if connection else None)
            if sock:
                sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if connection:
            connection.close()

    def expire():
        expired.set()
        close()

    try:
        if barrier:
            barrier.wait(timeout=min(5, timeout))
        row["start_ns"] = time.monotonic_ns()
        limit = min(timeout, run_deadline - time.monotonic())
        if limit <= 0:
            raise TimeoutError("overall run deadline exhausted")
        parsed, path = endpoint(url)
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = cls(parsed.hostname, parsed.port, timeout=limit)
        timer = threading.Timer(limit, expire)
        timer.daemon = True
        timer.start()
        connection.request("POST", path, case["encoded"],
                           {"Content-Type": "application/json", "X-Request-ID": request_id,
                            "User-Agent": "q36-unconstrained-isolation"})
        response = connection.getresponse()
        row["status"] = response.status
        live_socket[0] = connection.sock
        if live_socket[0] is None:
            try:
                live_socket[0] = response.fp.raw._sock
            except AttributeError:
                pass
        if response.status != 200:
            raise ValueError("HTTP " + str(response.status) + ": " + response.read(65536).decode(errors="replace"))
        if not response.getheader("Content-Type", "").lower().startswith("text/event-stream"):
            raise ValueError("expected text/event-stream")
        parse_sse(response, row, case["payload"]["max_tokens"])
    except Exception as error:
        row["errors"].append(type(error).__name__ + ": " + str(error))
    finally:
        if timer:
            timer.cancel()
        close()
        if expired.is_set():
            row["errors"].append("absolute request deadline exceeded")
        row["end_ns"] = time.monotonic_ns()
    return row


def compare(expected, actual):
    a, b = expected["text"], actual["text"]
    offset = next((index for index, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage_equal = all((expected.get("usage") or {}).get(key) == (actual.get("usage") or {}).get(key) for key in usage_keys)
    text_equal = a == b
    valid = not expected["errors"] and not actual["errors"] and expected["done"] and actual["done"]
    return {"case": actual["case"], "reference_request_id": expected["request_id"],
            "actual_request_id": actual["request_id"], "valid_streams": valid,
            "text_equal": text_equal, "finish_equal": expected["finish_reason"] == actual["finish_reason"],
            "usage_core_equal": usage_equal,
            "cache_usage_excluded_from_equality": True,
            "first_difference": None if text_equal else {"character_offset": offset,
                "reference_fragment": a[max(0, offset - 20):offset + 60],
                "actual_fragment": b[max(0, offset - 20):offset + 60],
                "reference_characters": len(a), "actual_characters": len(b)},
            "observable_exact_agreement": bool(valid and text_equal and usage_equal and expected["finish_reason"] == actual["finish_reason"])}


def run_suite(serial_url, parallel_url, model, output, callers=4, tokens=(64, 128),
              profiles=("greedy", "sampled"), timeout=30, total_timeout=600, parallel_model=None):
    endpoint(serial_url)
    endpoint(parallel_url)
    if callers not in (2, 4) or any(n not in (64, 128) for n in tokens) or any(p not in PROFILES for p in profiles):
        raise ValueError("invalid bounded suite configuration")
    if not 0 < timeout <= 120 or not 0 < total_timeout <= 1200:
        raise ValueError("invalid bounded deadline")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    end = time.monotonic() + total_timeout
    report = {"check": "unconstrained observable response repeatability and concurrency agreement",
              "limits": "No internal token IDs/logits/RNG-state/GPU overlap proof. Text differences are reported, not relabeled as isolation passes.",
              "native_trace_requirement": "Correlate X-Request-ID/time windows with native slot epochs and grouped-forward telemetry independently.",
              "callers": callers, "tokens": list(tokens), "profiles": list(profiles),
              "timeout_s": timeout, "total_timeout_s": total_timeout, "arms": [], "passed": False}

    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for profile in profiles:
        for budget in tokens:
            fixtures = cases(model, profile, budget)
            others = fixtures if not parallel_model or parallel_model == model else cases(parallel_model, profile, budget)
            arm = {"profile": profile, "max_tokens": budget,
                   "bodies": [case["payload"] for case in fixtures], "sequential": [], "concurrent_rounds": [],
                   "seed_repeat_comparisons": [], "concurrent_comparisons": []}
            report["arms"].append(arm)
            for case in fixtures:
                rows = []
                for repeat in range(2):
                    rid = f"q36-isolation-{nonce}-{profile}-{budget}-seq-{case['case']}-{repeat}"
                    row = request(serial_url, case, timeout, end, rid)
                    row.update(phase="sequential", repeat=repeat)
                    rows.append(row)
                    arm["sequential"].append(row)
                    save()
                arm["seed_repeat_comparisons"].append(compare(*rows))
            baseline = {row["case"]: row for row in arm["sequential"] if row["repeat"] == 0}
            for round_id, order in enumerate((others, list(reversed(others)))):
                round_rows = []
                for start in range(0, len(order), callers):
                    batch = order[start:start + callers]
                    barrier = threading.Barrier(len(batch))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
                        futures = [executor.submit(request, parallel_url, case, timeout, end,
                            f"q36-isolation-{nonce}-{profile}-{budget}-parallel-{round_id}-{case['case']}", barrier) for case in batch]
                        round_rows.extend(future.result(timeout=timeout + 10) for future in futures)
                round_rows.sort(key=lambda row: row["case"])
                arm["concurrent_rounds"].append(round_rows)
                arm["concurrent_comparisons"].extend(compare(baseline[row["case"]], row) for row in round_rows)
                save()
    comparisons = [row for arm in report["arms"] for key in ("seed_repeat_comparisons", "concurrent_comparisons") for row in arm[key]]
    report["summary"] = {"comparisons": len(comparisons),
        "valid_stream_pairs": sum(row["valid_streams"] for row in comparisons),
        "exact_text_agreements": sum(row["valid_streams"] and row["text_equal"] for row in comparisons),
        "observable_exact_agreements": sum(row["observable_exact_agreement"] for row in comparisons),
        "seed_repeat_agreements": sum(row["observable_exact_agreement"] for arm in report["arms"] for row in arm["seed_repeat_comparisons"]),
        "concurrent_agreements": sum(row["observable_exact_agreement"] for arm in report["arms"] for row in arm["concurrent_comparisons"])}
    report["passed"] = bool(comparisons) and all(row["observable_exact_agreement"] for row in comparisons)
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="use one already-running service for sequential and concurrent arms")
    parser.add_argument("--serial-url")
    parser.add_argument("--parallel-url")
    parser.add_argument("--model", required=True)
    parser.add_argument("--parallel-model")
    parser.add_argument("--callers", type=int, choices=(2, 4), default=4)
    parser.add_argument("--tokens", type=int, nargs="+", choices=(64, 128), default=[64, 128])
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--total-timeout", type=float, default=600)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print("Dry run: no HTTP calls, GPU operations or files. Add --run to test already-running services.")
        return 0
    serial = args.serial_url or args.url
    parallel = args.parallel_url or args.url
    if not serial or not parallel or args.out is None:
        parser.error("--run requires --out and --url (or both service URLs)")
    report = run_suite(serial, parallel, args.model, args.out, args.callers, tuple(args.tokens),
                       tuple(args.profiles), args.timeout, args.total_timeout, args.parallel_model)
    print(json.dumps({"passed": report["passed"], **report["summary"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
