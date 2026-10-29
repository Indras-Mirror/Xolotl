#!/usr/bin/env python3
"""Measure real live-KV suffixes on one native M31 worker; no hardware control.

Use the same IDs, resources and measurement window for each ABBA arm. Each
sample clears history, primes a fresh prefix, then measures only the append.
All requests, warmups, protocol rows and stderr diagnostics are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import statistics
import struct
import subprocess
import sys
import threading
import time


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def seconds(value):
    number = nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return number


def suffixes(value):
    try:
        numbers = [positive_int(part) for part in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError("expected positive comma-separated lengths") from None
    if len(set(numbers)) != len(numbers):
        raise argparse.ArgumentTypeError("suffix lengths must be unique")
    return numbers


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="first GGUF shard")
    parser.add_argument("--ids-file", required=True, type=Path, help="little-endian signed int32 text IDs")
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--prefix", type=positive_int, default=1024)
    parser.add_argument("--suffixes", type=suffixes, default=[100, 300])
    parser.add_argument("--repetitions", type=positive_int, default=3, help="measured repetitions after one warmup")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=positive_int, default=32)
    parser.add_argument("--context", type=positive_int, default=40000)
    parser.add_argument("--chunk", type=positive_int, default=32768)
    parser.add_argument("--arena-gib", type=nonnegative, default=0)
    parser.add_argument("--gpu-experts-gib", type=nonnegative, default=0)
    parser.add_argument("--register-arena", type=int, choices=(0, 1), default=0)
    parser.add_argument("--register-limit-gib", type=nonnegative, default=0)
    parser.add_argument("--pack-threads", type=positive_int, default=8)
    parser.add_argument("--startup-timeout", type=seconds, default=900)
    parser.add_argument("--request-timeout", type=seconds, default=7200)
    parser.add_argument("--shutdown-timeout", type=seconds, default=15)
    args = parser.parse_args(argv)
    if args.prefix % 256:
        parser.error("--prefix must be a multiple of the engine's 256-token reuse boundary")
    if args.prefix + max(args.suffixes) > args.context:
        parser.error("--context must cover prefix plus the largest suffix")
    if args.chunk > 32768 or args.threads > 512 or args.pack_threads > 16 or args.device < 0:
        parser.error("invalid chunk/thread/device resource bound")
    if args.register_arena and (args.backend != "cuda" or not args.register_limit_gib):
        parser.error("arena registration requires CUDA and a positive registration limit")
    if args.backend == "cpu" and args.gpu_experts_gib:
        parser.error("a GPU expert budget requires CUDA")
    return args


def load_ids(path, required):
    raw = Path(path).read_bytes()
    if len(raw) % 4 or len(raw) < 4 * required:
        raise ValueError(f"IDs must contain at least {required} little-endian int32 values with no trailing bytes")
    result = [value for (value,) in struct.iter_unpack("<i", raw)]
    if any(value < 0 for value in result):
        raise ValueError("text IDs must be nonnegative")
    return result


def engine_command(args):
    return [str(args.engine.resolve()), "--serve", "--model", str(args.model.resolve()),
            "--backend", args.backend, "--device", str(args.device), "--threads", str(args.threads),
            "--ctx", str(args.context), "--chunk", str(args.chunk), "--arena-gib", str(args.arena_gib),
            "--gpu-experts-gib", str(args.gpu_experts_gib), "--register-arena", str(args.register_arena),
            "--register-limit-gib", str(args.register_limit_gib), "--gpu-pack-threads", str(args.pack_threads)]


def finite(value, label):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("invalid " + label)
    return result


def validate_request(row, prompt, reused):
    """Reject cold fallbacks, partial reuse and incomplete/changing chunk traces."""
    done = row.get("done", {})
    expected_resume = [reused] if reused else []
    if (row.get("resume") != expected_resume or done.get("prompt") != prompt or
            done.get("reused") != reused or done.get("read") != prompt - reused or
            done.get("generated") != 1 or len(row.get("tokens", [])) != 1 or
            not isinstance(row["tokens"][0], int) or row["tokens"][0] < 0 or done.get("finish") != "length"):
        raise ValueError("wrong live-prefix contract: " + json.dumps(row))
    previous = reused
    for progress in row.get("pp", []):
        if (progress["total"] != prompt or not previous < progress["read"] <= prompt or
                progress["chunk"] != progress["read"] - previous):
            raise ValueError("invalid PP chunk trace")
        previous = progress["read"]
        finite(progress["ms"], "PP time")
        finite(progress["tps"], "PP rate")
    if previous != prompt:
        raise ValueError("incomplete PP chunk trace")
    finite(done["prompt_ms"], "prompt time")
    finite(done["decode_ms"], "decode time")
    return row


class Worker:
    def __init__(self, args, directory):
        self.args = args
        self.queue = queue.Queue()
        self.writers = []
        self.stdout_log = (directory / "stdout.log").open("x", encoding="utf-8")
        self.stderr_log = (directory / "stderr.log").open("x", encoding="utf-8")
        self.stdin_log = (directory / "stdin.log").open("x", encoding="utf-8")
        self.requests = (directory / "requests.jsonl").open("x", encoding="utf-8")
        self.process = None
        self.reader = None
        self.ready = []
        try:
            self.command = engine_command(args)
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr_log, text=True, encoding="utf-8", errors="replace", bufsize=1)
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            deadline = time.monotonic() + args.startup_timeout
            while True:
                line = self.next_line(deadline)
                self.ready.append(line)
                if line.startswith("ERR "):
                    raise RuntimeError("worker startup: " + line)
                if line.startswith("READY "):
                    if int(line.split()[1]) < args.prefix + max(args.suffixes):
                        raise ValueError("worker context does not cover prefix and suffix")
                    break
        except BaseException:
            self.close(graceful=False)
            raise

    def _read(self):
        try:
            for raw in self.process.stdout:
                self.stdout_log.write(raw)
                self.stdout_log.flush()
                self.queue.put(raw.rstrip("\r\n"))
        finally:
            self.queue.put(None)

    def next_line(self, deadline):
        try:
            line = self.queue.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            raise TimeoutError("worker output deadline; inspect stdout.log and stderr.log") from None
        if line is None:
            raise RuntimeError(f"worker exited: {self.process.poll()}; inspect stderr.log")
        return line

    def send(self, line, deadline):
        self.stdin_log.write(line + "\n")
        self.stdin_log.flush()
        result = queue.Queue(maxsize=1)
        def write():
            try:
                self.process.stdin.write(line + "\n")
                self.process.stdin.flush()
                result.put(None)
            except Exception as error:
                result.put(error)
        writer = threading.Thread(target=write, daemon=True)
        self.writers.append(writer)
        writer.start()
        try:
            error = result.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            raise TimeoutError("worker input deadline") from None
        if error:
            raise RuntimeError("worker input failed: " + str(error)) from error

    def generate(self, name, ids, cache):
        started = time.monotonic()
        row = {"case": name, "prompt": len(ids), "cache_prompt": cache,
               "ids_sha256": hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest(),
               "resume": [], "pp": [], "tokens": [], "raw": []}
        try:
            deadline = started + self.args.request_timeout
            self.send("GEN 1 temperature=0 seed=123 ignore_eos=1 cache_prompt=" + str(int(cache)) +
                      " " + ",".join(map(str, ids)), deadline)
            while True:
                line = self.next_line(deadline)
                row["raw"].append(line)
                fields = line.split()
                if fields and fields[0] == "RESUME" and len(fields) == 2:
                    row["resume"].append(int(fields[1]))
                elif fields and fields[0] == "PP" and len(fields) == 5:
                    previous = row["pp"][-1]["read"] if row["pp"] else (row["resume"][-1] if row["resume"] else 0)
                    row["pp"].append({"read": int(fields[1]), "total": int(fields[2]),
                        "ms": finite(fields[3], "PP time"), "tps": finite(fields[4], "PP rate"),
                        "chunk": int(fields[1]) - previous})
                elif fields and fields[0] == "T" and len(fields) == 2:
                    row["tokens"].append(int(fields[1]))
                elif fields and fields[0] == "DONE" and len(fields) >= 15:
                    row["done"] = {"generated": int(fields[1]), "prompt": int(fields[2]),
                        "prompt_ms": finite(fields[3], "prompt time"), "decode_ms": finite(fields[4], "decode time"),
                        "finish": fields[5], "reused": int(fields[8]), "read": int(fields[14])}
                    break
                else:
                    raise RuntimeError("unexpected worker record: " + line)
            return row
        except BaseException as error:
            row["error"] = str(error)
            raise
        finally:
            row["wall_seconds"] = time.monotonic() - started
            self.requests.write(json.dumps(row, allow_nan=False) + "\n")
            self.requests.flush()

    def close(self, graceful=True):
        if self.process and self.process.poll() is None:
            try:
                if graceful:
                    self.send("QUIT", time.monotonic() + self.args.shutdown_timeout)
                    self.process.wait(timeout=self.args.shutdown_timeout)
                else:
                    raise TimeoutError("failed request")
            except (OSError, RuntimeError, subprocess.TimeoutExpired, TimeoutError):
                self.process.terminate()
                try:
                    self.process.wait(timeout=self.args.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=self.args.shutdown_timeout)
        for thread in self.writers + ([self.reader] if self.reader else []):
            thread.join(timeout=self.args.shutdown_timeout)
        if self.process:
            self.process.stdin.close()
            self.process.stdout.close()
        for stream in (self.stdout_log, self.stderr_log, self.stdin_log, self.requests):
            stream.close()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    ids = load_ids(args.ids_file, args.prefix + max(args.suffixes))
    if not args.model.is_file() or not args.engine.is_file() or not os.access(args.engine, os.X_OK):
        raise ValueError("--model must name a GGUF file and --engine an executable")
    identities = {"engine_sha256": sha256(args.engine), "ids_sha256": sha256(args.ids_file)}
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"format": "m31-prefix-speed-v1", **identities, "complete": False,
        "argv": engine_command(args), "client_argv": list(sys.argv[1:] if argv is None else argv),
        "client_sha256": sha256(__file__), "ids_file": str(args.ids_file.resolve()), "ids_count": len(ids),
        "model": str(args.model.resolve()), "prefix": args.prefix, "suffixes": args.suffixes,
        "repetitions": args.repetitions, "warmup_repetitions": 1,
        "environment": {name: os.environ[name] for name in ("CUDA_VISIBLE_DEVICES", "GGML_CUDA_DISABLE_FUSION",
            "GGML_CUDA_MMQ_PREC", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS") if name in os.environ},
        "sampling": {"max_tokens": 1, "temperature": 0, "seed": 123, "ignore_eos": True},
        "timing_contract": "append wall and DONE prompt_ms only; reset and prime excluded; prompt_ms includes first output head/sample",
        "runs": []}
    worker = None
    try:
        save(args.out / "receipt.json", receipt)
        worker = Worker(args, args.out)
        receipt["startup"] = worker.ready
        for repetition in range(args.repetitions + 1):
            for suffix in args.suffixes:
                name = f"{repetition:02d}-{suffix}"
                row = {"repetition": repetition, "suffix": suffix, "warmup": repetition == 0}
                receipt["runs"].append(row)
                row["reset"] = validate_request(worker.generate(name + "-reset", ids[:2], False), 2, 0)
                row["prime"] = validate_request(worker.generate(name + "-prime", ids[:args.prefix], True), args.prefix, 0)
                row["append"] = validate_request(worker.generate(name + "-append", ids[:args.prefix + suffix], True),
                                                 args.prefix + suffix, args.prefix)
                save(args.out / "receipt.json", receipt)
        receipt["summary"] = {str(suffix): {"samples": args.repetitions,
            "median_prompt_ms": statistics.median(row["append"]["done"]["prompt_ms"] for row in receipt["runs"]
                if row["suffix"] == suffix and not row["warmup"]),
            "median_wall_seconds": statistics.median(row["append"]["wall_seconds"] for row in receipt["runs"]
                if row["suffix"] == suffix and not row["warmup"])} for suffix in args.suffixes}
        receipt["complete"] = True
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], file=sys.stderr)
    finally:
        if worker:
            worker.close(graceful=receipt["complete"])
            receipt["worker_exit"] = worker.process.returncode
        receipt["final_engine_sha256"] = sha256(args.engine)
        receipt["final_ids_sha256"] = sha256(args.ids_file)
        receipt["unchanged"] = all(receipt["final_" + name] == value for name, value in identities.items())
        receipt["pass"] = receipt["complete"] and receipt["unchanged"] and receipt.get("worker_exit") == 0
        save(args.out / "receipt.json", receipt)
    print(json.dumps({"complete": receipt["complete"], "unchanged": receipt["unchanged"],
                      "summary": receipt.get("summary", {})}, indent=2))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
