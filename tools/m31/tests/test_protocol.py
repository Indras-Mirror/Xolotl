#!/usr/bin/env python3
"""Production line-loop regressions with a deterministic model-free engine."""
import argparse
import queue
import subprocess
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from serve import server as S


class Engine(unittest.TestCase):
    def setUp(self):
        self.p = subprocess.Popen([EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lines = queue.Queue()
        def pump():
            for line in self.p.stdout:
                self.lines.put(line.rstrip())
            self.lines.put(None)
        self.reader = threading.Thread(target=pump, daemon=True)
        self.reader.start()
        self.startup = self.until("READY")

    def tearDown(self):
        if self.p.poll() is None:
            if not self.p.stdin.closed:
                self.send("QUIT")
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait(timeout=5)
        self.assertEqual(self.p.returncode, 0, self.p.stderr.read())
        if not self.p.stdin.closed:
            self.p.stdin.close()
        self.p.stdout.close()
        self.p.stderr.close()

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def until(self, prefix):
        lines = []
        while True:
            line = self.lines.get(timeout=5)
            self.assertIsNotNone(line, "unexpected engine exit")
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def request(self, command):
        self.send(command)
        return self.until("DONE")


class Protocol(Engine):
    def test_zero_tokens_and_rejection_recovery(self):
        lines = self.request("GEN 0 1,2")
        self.assertFalse(any(line.startswith("T ") for line in lines))
        self.assertEqual(lines[-1].split()[1], "0")
        for bad in ("GEN -1 1", "GEN 1 unknown=3 1", "GEN 1 chat_b64=broken 1",
                    "GENI 1 ignored 1", "BGENI 3 1 ignored 1"):
            self.send(bad)
            self.until("ERR")
            self.assertIn("T 3", self.request("GEN 1 1,2"))

    def test_vision_wire_one_prefill_and_recovery(self):
        if not any("vision=1" in line for line in self.startup):
            self.skipTest("STRATA_M31_VISION is off")
        with tempfile.TemporaryDirectory(prefix="m31-protocol-") as directory:
            blob = Path(directory) / "input.sve1"
            blob.write_bytes(struct.pack("<5i8f", 0x31455653,2,2,1,4,*([.25]*8)))
            lines = self.request(f"GENI 1 {blob} 1,-1,-1,2")
            self.assertIn("T 3", lines)
            self.assertEqual(len([line for line in lines if line.startswith("PP ")]), 1)
            self.assertEqual(lines[-1].split()[2], "4")
            batched = self.request(f"BGENI 3 1 {blob} 1,-1,-1,2")
            self.assertIn("T 3", batched)
            self.assertEqual(self.until("BADM")[-1], "BADM 3 0")
            for bad in (f"{blob} 1,-1,2,-1", f"{blob}-absent 1,-1,-1,2"):
                self.send("GENI 1 " + bad)
                self.assertIn("vision unavailable", self.until("ERR")[-1])
                self.assertIn("T 3", self.request("GEN 1 1,2"))
            blob.write_bytes(struct.pack("<5i8f",0x31455653,2,2,1,4,float("nan"),*([.25]*7)))
            self.send(f"GENI 1 {blob} 1,-1,-1,2")
            self.assertIn("non-finite",self.until("ERR")[-1])
            self.assertIn("T 3", self.request("GEN 1 1,2"))

    def test_done_uses_common_parser_fields_and_milliseconds(self):
        line = self.request("GEN 2 1,2,3")[-1]
        self.assertEqual(len(line.split()), 16)
        engine = SimpleNamespace()
        S.StrataEngine._parse_done(engine, line)
        self.assertEqual(engine.last["generated"], 2)
        self.assertEqual(engine.last["prompt_tokens"], 3)
        self.assertEqual(engine.last["prompt_read"], 3)
        self.assertEqual(engine.last["reused"], 0)
        self.assertEqual(engine.last["finish"], "length")
        self.assertGreater(engine.last["prompt_ms"], 1)
        self.assertGreater(engine.last["decode_ms"], 1)
        self.assertLess(engine.last["prompt_ms"], 5000)
        self.assertEqual(engine.last["offloaded"], 0)

    def test_first_sample_is_prompt_work_and_decode_has_n_minus_one_steps(self):
        single = self.request("GEN 1 1,2,3")[-1].split()
        self.assertEqual(single[1], "1")
        self.assertGreater(float(single[3]), 0)
        self.assertEqual(float(single[4]), 0)
        empty = self.request("GEN 0 1,2,3")[-1].split()
        self.assertEqual(empty[1], "0")
        self.assertEqual(float(empty[4]), 0)
        stopped = self.request("GEN 5 logit_bias=7:100 1,2,3")[-1].split()
        self.assertEqual(stopped[1], "1")
        self.assertEqual(stopped[5], "stop")
        self.assertEqual(float(stopped[4]), 0)

    def test_nonfinite_and_grammar_mask_recovery(self):
        nonfinite = self.request("GEN 2 6,1")[-1].split()
        self.assertEqual(nonfinite[5], "nonfinite")
        self.assertIn("T 3", self.request("GEN 1 1,2"))
        constrained = self.request("GEN 2 chat_b64=allow2 1,2")
        self.assertEqual([line for line in constrained if line.startswith("T ")], ["T 2", "T 2"])
        self.send("GEN 1 allowed_ids=4 chat_b64=allow2 1,2")
        self.assertIn("empty sampling support", self.until("ERR")[-1])
        self.assertIn("T 3", self.request("GEN 1 1,2"))

    def test_prefix_branch_cache_opt_out(self):
        ids = [1] * 600
        cold = self.request("GEN 1 " + ",".join(map(str, ids)))
        self.assertFalse(any(line.startswith("RESUME") for line in cold))
        warm = self.request("GEN 1 " + ",".join(map(str, ids + [2])))
        self.assertIn("RESUME 512", warm)
        branch = ids.copy()
        branch[300] = 2
        edited = self.request("GEN 1 " + ",".join(map(str, branch)))
        self.assertIn("RESUME 256", edited)
        uncached = self.request("GEN 1 cache_prompt=0 " + ",".join(map(str, branch)))
        self.assertFalse(any(line.startswith("RESUME") for line in uncached))
        after = self.request("GEN 1 " + ",".join(map(str, branch)))
        self.assertFalse(any(line.startswith("RESUME") for line in after))

    def test_stop_during_prefill_and_decode(self):
        self.send("GEN 10 " + ",".join(["1"] * 3000))
        self.until("PP")
        self.send("STOP")
        lines = self.until("DONE")
        self.assertEqual(lines[-1].split()[5], "cancel")
        self.assertLess(int(lines[-1].split()[14]), 3000)
        retried = self.request("GEN 1 " + ",".join(["1"] * 3000))
        self.assertFalse(any(line.startswith("RESUME") for line in retried))
        self.assertIn("T 3", self.request("GEN 1 1,2"))
        self.send("GEN 100 1,2")
        self.until("T")
        self.send("STOP")
        self.assertEqual(self.until("DONE")[-1].split()[5], "cancel")
        self.assertIn("T 3", self.request("GEN 1 1,2"))

    def test_quit_during_prefill_joins_reader(self):
        self.send("GEN 10 " + ",".join(["1"] * 3000))
        self.until("PP")
        self.send("QUIT")
        self.assertEqual(self.until("DONE")[-1].split()[5], "cancel")
        self.p.wait(timeout=5)

    def test_eof_during_prefill_and_decode(self):
        self.send("GEN 10 " + ",".join(["1"] * 3000))
        self.until("PP")
        self.p.stdin.close()
        finished = self.until("DONE")[-1].split()
        self.assertEqual(finished[5], "cancel")
        self.assertLess(int(finished[14]), 3000)
        self.p.wait(timeout=5)

    def test_eof_during_decode(self):
        self.send("GEN 1000 1,2")
        self.until("T")
        self.p.stdin.close()
        finished = self.until("DONE")[-1].split()
        self.assertEqual(finished[5], "cancel")
        self.assertLess(int(finished[1]), 1000)
        self.p.wait(timeout=5)

    def test_seed_is_scoped_to_generation(self):
        request = "GEN 20 temperature=1 seed=0 ignore_eos=1 1,2"
        first = self.request(request)
        self.request("GEN 5 temperature=1 seed=18446744073709551615 ignore_eos=1 1,2")
        again = self.request(request)
        self.assertEqual([line for line in first if line.startswith("T ")],
                         [line for line in again if line.startswith("T ")])

    def test_ignore_eos_and_explicit_stop(self):
        stopped = self.request("GEN 5 logit_bias=7:100 1,2")
        self.assertEqual([line for line in stopped if line.startswith("T ")], ["T 7"])
        self.assertEqual(stopped[-1].split()[5], "stop")
        ignored = self.request("GEN 3 ignore_eos=1 logit_bias=7:100 1,2")
        self.assertEqual([line for line in ignored if line.startswith("T ")], ["T 3"] * 3)
        self.assertEqual(ignored[-1].split()[5], "length")
        explicit = self.request("GEN 5 ignore_eos=1 stop=3 1,2")
        self.assertEqual(explicit[-1].split()[1], "1")
        self.assertEqual(explicit[-1].split()[5], "stop")

    def test_sleep_wake_keeps_conversation(self):
        ids = [1] * 600
        self.request("GEN 1 " + ",".join(map(str, ids)))
        self.send("SLEEP")
        slept = self.until("SLEPT")[-1].split()
        self.assertIn("kv_tokens=600", slept)
        self.assertIn("conversation=1", slept)
        self.send("SLEEP")                     # idempotent while asleep
        self.assertIn("kv_tokens=600", self.until("SLEPT")[-1].split())
        self.send("WAKE")
        woke = self.until("READY")
        self.assertTrue(woke[0].startswith("INFO woke=1"), woke)
        self.assertEqual(woke[-1], "READY 4096 stop")
        self.send("WAKE")                      # already awake: READY only
        self.assertEqual(self.until("READY"), ["READY 4096 stop"])
        warm = self.request("GEN 1 " + ",".join(map(str, ids + [2])))
        self.assertIn("RESUME 512", warm)

    def test_request_wakes_a_sleeping_engine(self):
        ids = [1] * 600
        self.request("GEN 1 " + ",".join(map(str, ids)))
        self.send("SLEEP")
        self.until("SLEPT")
        lines = self.request("GEN 1 " + ",".join(map(str, ids + [2])))
        self.assertTrue(lines[0].startswith("INFO woke=1"), lines)
        self.assertIn("RESUME 512", lines)
        self.assertIn("T 3", lines)

    def test_batch_completion_and_mask(self):
        lines = self.request("BGEN 3 2 allowed_ids=2,4 logit_bias=4:10 1,2")
        self.assertEqual([line for line in lines if line.startswith("T ")], ["T 4", "T 4"])
        self.assertEqual(self.until("BADM")[-1], "BADM 3 0")


class FailedWake(Engine):
    def setUp(self):
        import os
        os.environ["M31_PROTOCOL_FIXTURE_WAKE_FAILURES"] = "1"
        try:
            super().setUp()
        finally:
            del os.environ["M31_PROTOCOL_FIXTURE_WAKE_FAILURES"]

    def test_failed_wake_stays_asleep_then_recovers(self):
        self.send("SLEEP")
        self.until("SLEPT")
        self.send("GEN 1 1,2")
        self.assertIn("still asleep", self.until("ERR")[-1])
        lines = self.request("GEN 1 1,2")
        self.assertTrue(lines[0].startswith("INFO woke=1"), lines)
        self.assertIn("T 3", lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    args, rest = parser.parse_known_args()
    EXE = args.fixture
    unittest.main(argv=[__file__, *rest])
