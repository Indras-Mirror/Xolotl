#!/usr/bin/env python3
"""Small CPU checks for numerical receipts and the portable quality client's guards."""
import json
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import compare_logits as C
import quality as Q
import replay_oracle as R
import speed as S
import prefix_speed as P
import copy
import hashlib
import struct
import time
import numpy as np


class BenchChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"
        for root in (self.reference, self.candidate):
            case = root / "case"
            case.mkdir(parents=True)
            data = {"format": "m31-logits-v1", "kv": "f16", "tokens": 2, "vocab": 2,
                "positions": [0, 1], "model": "fixture", "actual_chunks": [{"position": 0, "tokens": 2}],
                "engine": "llama.cpp-PR31192" if root == self.reference else "xolotl-m31",
                "oracle_op_offload": root == self.reference, "cuda_fusion_disabled": True,
                "context": 4096, "head_tail": 0}
            (case / "manifest.json").write_text(json.dumps(data))
            np.asarray([1, 2], dtype="<i4").tofile(case / "tokens.i32")
            np.asarray([[0., 1.], [1., 0.]], dtype="<f4").tofile(case / "logits.f32")

    def modify(self, **updates):
        path = self.candidate / "case/manifest.json"
        data = json.loads(path.read_text())
        data.update(updates)
        path.write_text(json.dumps(data))

    def test_bitwise_is_distinct_from_zero_numeric_error(self):
        result = C.compare_suite(self.reference, self.candidate, require_bitwise=True)
        self.assertTrue(result["pass"])
        np.asarray([[-0., 1.], [1., 0.]], dtype="<f4").tofile(self.candidate / "case/logits.f32")
        result = C.compare_suite(self.reference, self.candidate, require_bitwise=True)
        self.assertEqual(result["mean_kl"], 0)
        self.assertFalse(result["bitwise_identical"])
        self.assertFalse(result["pass"])

    def test_chunk_mismatch_is_refused(self):
        self.modify(actual_chunks=[{"position": 0, "tokens": 1}, {"position": 1, "tokens": 1}])
        with self.assertRaisesRegex(ValueError, "chunk geometry"):
            C.compare_suite(self.reference, self.candidate)

    def test_cpu_oracle_cannot_impersonate_gpu(self):
        path = self.reference / "case/manifest.json"
        data = json.loads(path.read_text())
        data["oracle_op_offload"] = False
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "placement"):
            C.compare_suite(self.reference, self.candidate)

    def test_truncated_rows_are_refused(self):
        np.asarray([0., 1.], dtype="<f4").tofile(self.candidate / "case/logits.f32")
        with self.assertRaisesRegex(ValueError, "truncated"):
            C.compare_suite(self.reference, self.candidate)

    def test_quality_guards(self):
        choice = {"finish_reason": "length", "message": {"content": "<think>leak", "tool_calls": [
            {"id": "same"}, {"id": "same"}]}}
        inspected = Q.inspect_choice(choice)
        self.assertTrue(inspected["marker_leak"])
        self.assertTrue(inspected["invalid_tool_ids"])
        self.assertEqual(inspected["finish"], "length")
        self.assertEqual(Q.number("$1,234.5."), 1234.5)
        for marker in ("<mm:think>", "</mm:think>", "]<]minimax[>[", "]~b]", "[e~["):
            self.assertTrue(Q.inspect_choice({"message": {"content": marker}})["marker_leak"])
        self.assertEqual(len(Q.CASES), 20)
        self.assertEqual(len(Q.FOLLOW), 3)

    def test_quality_wire_sampling_is_explicit(self):
        args = SimpleNamespace(model="minimax-m31", url="http://127.0.0.1:1", api_key="", timeout=1)
        evaluation = Q.Evaluation(args, io.StringIO(), io.StringIO())
        captured = []
        def response(request, timeout):
            captured.append(json.loads(request.data))
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())
        with patch("urllib.request.urlopen", side_effect=response):
            evaluation.chat([{"role": "user", "content": "fixture"}])
        body = captured[0]
        self.assertEqual({k: body[k] for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens")},
                         {"temperature": 1.0, "top_p": .95, "top_k": 40, "min_p": .05, "max_tokens": 6000})
        for key in ("seed", "reasoning_effort", "chat_template_kwargs"):
            self.assertNotIn(key, body)

    def test_quality_seed_zero_and_max_uint64_are_sent_exactly(self):
        for seed in (0, (1 << 64) - 1):
            with self.subTest(seed=seed):
                args = SimpleNamespace(model="minimax-m31", url="http://127.0.0.1:1", api_key="", timeout=1, seed=seed)
                raw = io.StringIO()
                evaluation = Q.Evaluation(args, io.StringIO(), raw)
                captured = []
                def response(request, timeout):
                    captured.append(json.loads(request.data))
                    return io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())
                with patch("urllib.request.urlopen", side_effect=response):
                    evaluation.chat([{"role": "user", "content": "fixture"}])
                self.assertEqual(captured[0]["seed"], seed)
                self.assertIs(type(captured[0]["seed"]), int)
                self.assertEqual(json.loads(raw.getvalue())["request"]["seed"], seed)

    def test_quality_cli_rejects_invalid_seed_before_io(self):
        script = Path(__file__).resolve().parents[1] / "bench/quality.py"
        for index, seed in enumerate(("-1", str(1 << 64), "1.5")):
            with self.subTest(seed=seed):
                output = self.root / f"invalid-seed-{index}"
                result = subprocess.run([sys.executable, str(script), "--suite", "tools23", "--out", str(output), "--seed", seed],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("seed", result.stderr)
                self.assertFalse(output.exists())

    def test_quality_summary_records_only_explicit_seed(self):
        for seed in (None, 0):
            with self.subTest(seed=seed):
                output = self.root / ("default-seed" if seed is None else "explicit-zero")
                argv = ["quality.py", "--suite", "tools23", "--out", str(output)]
                if seed is not None:
                    argv += ["--seed", str(seed)]
                def response(request, timeout):
                    return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "fixture"}}]}).encode())
                with patch.object(sys, "argv", argv), patch("urllib.request.urlopen", side_effect=response), redirect_stdout(io.StringIO()):
                    result = Q.main()
                self.assertEqual(result, 1)  # Synthetic replies fail the original tool predicates.
                summary = json.loads((output / "summary.json").read_text())
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["tools23"]["total"], 23)
                self.assertFalse(summary["pass"])
                self.assertEqual("seed" in summary["sampling"], seed is not None)
                if seed is not None:
                    self.assertEqual(summary["sampling"]["seed"], seed)
                requests = [json.loads(line)["request"] for line in (output / "raw.jsonl").read_text().splitlines()]
                self.assertEqual(len(requests), 23)
                self.assertTrue(all(("seed" in body) == (seed is not None) for body in requests))

    def test_speed_uses_decode_transitions_for_both_servers(self):
        # Deliberately inconsistent API rates must not change either denominator.
        common = {"timings": {"prompt_n": 1000, "prompt_ms": 500, "predicted_n": 256,
            "predicted_ms": 25500, "prompt_per_second": 9999, "predicted_per_second": 256 / 25.5}}
        reference = {"timings": {**common["timings"], "predicted_per_second": 10}}
        a, b = S.timing_metrics(common), S.timing_metrics(reference)
        self.assertEqual(a["decode_steps"], 255)
        self.assertEqual(a["gen_tps"], 10)
        self.assertEqual(a["gen_tps"], b["gen_tps"])
        self.assertEqual(a["prefill_tps"], 2000)
        self.assertNotEqual(a["raw_gen_tps"], b["raw_gen_tps"])
        self.assertEqual(a["raw_prefill_tps"], 9999)

    def test_speed_single_token_has_no_decode_transition(self):
        response = {"timings": {"prompt_n": 1, "prompt_ms": 10, "predicted_n": 1, "predicted_ms": 0}}
        self.assertEqual(S.timing_metrics(response)["gen_tps"], 0)
        response["timings"]["predicted_n"] = 2
        with self.assertRaisesRegex(ValueError, "zero time"):
            S.timing_metrics(response)

    def test_public_long_generator_is_complete_and_deterministic(self):
        script = Path(__file__).resolve().parents[1] / "bench/make_prompts.py"
        first, second = self.root / "long-first", self.root / "long-second"
        for output in (first, second):
            subprocess.run([sys.executable, str(script), str(output), "--suite", "long-public"], check=True,
                           capture_output=True, text=True, timeout=10)
        manifest = json.loads((first / "suite.json").read_text())
        self.assertEqual({(r["group"], r["tokens"]) for r in manifest},
                         {("ru", 16384), ("ru", 32768), ("code", 16384), ("code", 32768)})
        self.assertEqual(len(manifest), 4)
        for row in manifest:
            self.assertGreaterEqual(row["passages"], 1600)
            name = row["case"]
            self.assertEqual(int((first / (name + ".length")).read_text()), row["tokens"])
            text = (first / (name + ".txt")).read_text(encoding="utf-8")
            self.assertGreater(len(text), 32768)
            self.assertIn("0", text)
            self.assertIn("2047", text)
            self.assertEqual((first / (name + ".txt")).read_bytes(), (second / (name + ".txt")).read_bytes())
        self.assertEqual({p.name for p in first.iterdir()}, {p.name for p in second.iterdir()})
        self.assertFalse(list(first.glob("*.offset")))

    def test_cache_observatory_has_published_partitions(self):
        script = Path(__file__).resolve().parents[1] / "bench/make_prompts.py"
        output = self.root / "observatory"
        subprocess.run([sys.executable, str(script), str(output), "--suite", "cache-observatory"],
                       check=True, capture_output=True, text=True, timeout=10)
        manifest = json.loads((output / "suite.json").read_text())
        self.assertEqual(len(manifest), 2)
        self.assertEqual({(r["tokens"], r["reuse"]) for r in manifest}, {(257, 256), (1324, 1024)})
        text_files = []
        for row in manifest:
            self.assertEqual(row["whole_chunks"], [row["tokens"]])
            self.assertEqual(row["split_chunks"], [row["reuse"], row["tokens"] - row["reuse"]])
            self.assertEqual(int((output / (row["case"] + ".length")).read_text()), row["tokens"])
            text_files.append((output / (row["case"] + ".txt")).read_text())
        self.assertEqual(text_files[0], text_files[1])
        self.assertTrue(text_files[0].startswith("Record 0: The observatory measured clear skies over the northern river. "
            "Observer number 0 entered the temperature and checked the previous report.\nRecord 1:"))
        self.assertEqual(len(text_files[0].splitlines()), 7000)
        self.assertTrue(text_files[0].endswith("Observer number 41 entered the temperature and checked the previous report."))

    def test_bounded_capacity_requires_trace(self):
        self.modify(context=40000)
        path = self.reference / "case/manifest.json"
        data = json.loads(path.read_text())
        data["context"] = 2
        path.write_text(json.dumps(data))
        trace = {"position": 0, "tokens": 2, "seq_min": 0, "seq_max": 1,
            "active_keys_padded_from_verified_positions": 256, "physical_context_requested": 2,
            "physical_context_actual": 256, "rope_ext_factor": 0, "rope_orig_context": 1048576,
            "rope_frequency_scale": 1}
        trace_path = self.reference / "case/capacity-equivalence.jsonl"
        trace_path.write_text(json.dumps(trace) + "\n")
        R.validate_capacity(self.reference, self.candidate, "case")
        trace["seq_min"] = 1
        trace_path.write_text(json.dumps(trace) + "\n")
        with self.assertRaisesRegex(ValueError, "active KV"):
            R.validate_capacity(self.reference, self.candidate, "case")


class PrefixSpeedChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model.gguf"
        self.model.write_text("model-free fixture")
        self.ids = self.root / "tokens.i32"
        self.ids.write_bytes(struct.pack("<1324i", *range(1324)))
        self.output = self.root / "results"
        self.argv = ["--engine", sys.executable, "--model", str(self.model),
                     "--ids-file", str(self.ids), "--out", str(self.output), "--repetitions", "1",
                     "--startup-timeout", "2", "--request-timeout", "2", "--shutdown-timeout", ".2"]

    def fake_worker(self, mode="normal"):
        script = self.root / "worker.py"
        script.write_text("MODE = " + repr(mode) + "\n" + r'''import sys, time
if MODE == "startup-timeout":
    time.sleep(60)
print("READY 40000 stop", flush=True)
history = []
for line in sys.stdin:
    if line.strip() == "QUIT":
        break
    if MODE == "reply-timeout":
        time.sleep(60)
    if MODE == "died":
        sys.exit(7)
    fields = line.split()
    assert fields[:2] == ["GEN", "1"]
    assert "temperature=0" in fields and "ignore_eos=1" in fields and "seed=123" in fields
    ids = list(map(int, fields[-1].split(",")))
    cache = "cache_prompt=1" in fields
    reused = 0
    if cache:
        for a, b in zip(history, ids):
            if a != b:
                break
            reused += 1
        reused = min(reused, len(ids) - 1) // 256 * 256
    if MODE == "cold-fallback" and reused:
        reused = 0
    if reused:
        print("RESUME", reused, flush=True)
    remaining = len(ids) - reused
    chunks = [remaining // 2, remaining - remaining // 2] if remaining > 2 else [remaining]
    read = reused
    for chunk in chunks:
        read += chunk
        print("PP", read, len(ids), "1.25 100", flush=True)
    print("T 42", flush=True)
    print("DONE", 1, len(ids), "2.5 0 length 0 0", reused, "0 0 0 0 0", remaining, 0, flush=True)
    history = ids if cache else []
''')
        return [sys.executable, str(script)]

    def run_fake(self, mode="normal"):
        with patch.object(P, "engine_command", return_value=self.fake_worker(mode)), redirect_stdout(io.StringIO()), \
                patch("sys.stderr", new=io.StringIO()):
            result = P.main(self.argv)
        return result, json.loads((self.output / "receipt.json").read_text())

    def test_cli_invalid_recipe_is_rejected_before_files_or_process(self):
        script = Path(P.__file__)
        for flag, value in (("--prefix", "1025"), ("--suffixes", "100,100"),
                            ("--repetitions", "0"), ("--request-timeout", "nan"),
                            ("--context", "1024"), ("--device", "-1")):
            with self.subTest(flag=flag):
                result = subprocess.run([sys.executable, str(script), *self.argv, flag, value],
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.output.exists())

    def test_input_is_little_endian_int32_and_must_cover_largest_suffix(self):
        self.assertEqual(P.load_ids(self.ids, 1324)[-1], 1323)
        self.ids.write_bytes(struct.pack("<1323i", *range(1323)))
        with self.assertRaisesRegex(ValueError, "at least 1324"):
            P.load_ids(self.ids, 1324)
        self.ids.write_bytes(struct.pack("<1324i", *range(1324)) + b"x")
        with self.assertRaisesRegex(ValueError, "trailing bytes"):
            P.load_ids(self.ids, 1324)

    def test_live_prefix_receipt_retains_warmups_raw_rows_and_actual_chunks(self):
        result, receipt = self.run_fake()
        self.assertEqual(result, 0)
        self.assertTrue(receipt["pass"])
        self.assertEqual(receipt["engine_sha256"], hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest())
        self.assertEqual(receipt["ids_sha256"], hashlib.sha256(self.ids.read_bytes()).hexdigest())
        self.assertEqual(receipt["argv"], [sys.executable, str(self.root / "worker.py")])
        self.assertEqual(len(receipt["runs"]), 4)  # Two suffixes, warmup plus one measured repetition.
        self.assertEqual(sum(row["warmup"] for row in receipt["runs"]), 2)
        for row in receipt["runs"]:
            self.assertEqual(row["reset"]["cache_prompt"], False)
            self.assertEqual(row["reset"]["done"]["read"], 2)
            self.assertEqual(row["prime"]["done"]["reused"], 0)
            self.assertEqual(row["append"]["resume"], [1024])
            self.assertEqual([p["chunk"] for p in row["append"]["pp"]], [row["suffix"] // 2] * 2)
            self.assertGreaterEqual(row["append"]["wall_seconds"], 0)
            self.assertEqual(row["append"]["done"]["prompt_ms"], 2.5)
        self.assertEqual(receipt["summary"]["100"]["samples"], 1)
        self.assertEqual(len((self.output / "requests.jsonl").read_text().splitlines()), 12)
        self.assertIn("RESUME 1024", (self.output / "stdout.log").read_text())
        self.assertIn("GEN 1 temperature=0", (self.output / "stdin.log").read_text())

    def test_cold_fallback_is_a_failure_and_diagnostics_survive(self):
        result, receipt = self.run_fake("cold-fallback")
        self.assertEqual(result, 1)
        self.assertFalse(receipt["pass"])
        self.assertIn("live-prefix contract", receipt["error"])
        self.assertTrue((self.output / "stderr.log").is_file())
        self.assertEqual(len((self.output / "requests.jsonl").read_text().splitlines()), 3)

    def test_startup_reply_and_dead_worker_are_bounded_failures(self):
        for index, mode in enumerate(("startup-timeout", "reply-timeout", "died")):
            with self.subTest(mode=mode):
                self.output = self.root / f"failure-{index}"
                argv = list(self.argv)
                argv[argv.index("--out") + 1] = str(self.output)
                argv[argv.index("--startup-timeout") + 1] = ".2" if mode == "startup-timeout" else "2"
                argv[argv.index("--request-timeout") + 1] = ".2"
                with patch.object(P, "engine_command", return_value=self.fake_worker(mode)), \
                        redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                    started = time.monotonic()
                    result = P.main(argv)
                self.assertLess(time.monotonic() - started, 3)
                self.assertEqual(result, 1)
                receipt = json.loads((self.output / "receipt.json").read_text())
                self.assertFalse(receipt["pass"])
                self.assertTrue(receipt["error"])
                self.assertTrue((self.output / "stdout.log").is_file())
                if mode != "startup-timeout":
                    self.assertIn("READY 40000", (self.output / "stdout.log").read_text())
                    requests = [json.loads(line) for line in (self.output / "requests.jsonl").read_text().splitlines()]
                    self.assertEqual(len(requests), 1)
                    self.assertIn("exited" if mode == "died" else "output deadline", requests[0]["error"])

    def test_receipt_validation_rejects_partial_resume_bad_chunks_and_nonfinite_time(self):
        _, receipt = self.run_fake()
        valid = receipt["runs"][0]["append"]
        for change in (lambda row: row.update(resume=[768]),
                       lambda row: row["done"].update(read=99),
                       lambda row: row["done"].update(generated=0),
                       lambda row: row["pp"][0].update(chunk=51),
                       lambda row: row["done"].update(prompt_ms=float("inf"))):
            with self.subTest(change=change):
                row = copy.deepcopy(valid)
                change(row)
                with self.assertRaises(ValueError):
                    P.validate_request(row, 1124, 1024)


if __name__ == "__main__":
    unittest.main()
