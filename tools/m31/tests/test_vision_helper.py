"""Exercise the actual optional helper transport, with no model or GPU."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))
from vision_encoder import VisionEncoder, VisionUnavailable


FAKE = """#!{python}
import json, os, pathlib, sys, time
mode = {mode!r}
if mode == 'startup_exit':
    sys.exit(7)
if mode == 'startup_partial':
    sys.stdout.write('{{'); sys.stdout.flush(); time.sleep(30)
if mode == 'startup_invalid':
    print('not JSON', flush=True); time.sleep(30)
if mode == 'startup_timeout':
    time.sleep(30)
if mode == 'stderr':
    sys.stderr.write('x' * 2000000); sys.stderr.flush()
print(json.dumps({{'ready': True}}), flush=True)
if mode == 'write_timeout':
    time.sleep(30)
for line in sys.stdin:
    request = json.loads(line)
    if mode == 'killed':
        os.kill(os.getpid(), 9)
    if mode == 'reply_partial':
        sys.stdout.write('{{'); sys.stdout.flush(); time.sleep(30)
    if mode == 'reply_timeout':
        time.sleep(30)
    if mode == 'reply_invalid':
        print('not JSON', flush=True); continue
    if mode == 'reply_error':
        print(json.dumps({{'ok': False, 'error': 'encoder failed'}}), flush=True); continue
    out = pathlib.Path(request['out']); out.mkdir()
    manifest = out / 'manifest.json'; manifest.write_text('{{}}')
    print(json.dumps({{'ok': True, 'manifest': str(manifest), 'rows': 1, 'visual_rows': 1}}), flush=True)
"""


@unittest.skipUnless(os.name == "posix", "nonblocking pipe transport is validated on Linux")
class HelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.adapter = self.root / "helper"
        self.model = self.root / "model.gguf"
        self.mmproj = self.root / "mmproj.gguf"
        self.media = self.root / "media.png"
        for path in (self.model, self.mmproj, self.media):
            path.write_bytes(b"test fixture")
        self.client = VisionEncoder(self.adapter, self.model, self.mmproj, self.root / "outputs",
                                    timeout=0.15, startup_timeout=0.15)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def helper(self, mode="ok"):
        self.adapter.write_text(FAKE.format(python=sys.executable, mode=mode))
        self.adapter.chmod(0o700)

    def encode(self, prompt="<__media__>"):
        return self.client.encode(prompt, [self.media])

    def assert_unavailable_and_recover(self, operation=None):
        start = time.monotonic()
        with self.assertRaisesRegex(VisionUnavailable, "^vision unavailable:"):
            (operation or self.encode)()
        self.assertLess(time.monotonic()-start, 4)
        self.assertIsNone(self.client.process)
        self.assertEqual(list((self.root / "outputs").glob("worker-*.json")), [])
        self.helper()
        self.assertTrue(self.encode().is_file())
        self.client.close()
        self.assertEqual(list((self.root / "outputs").glob("worker-*.json")), [])

    def test_missing_binary(self):
        self.assert_unavailable_and_recover()

    def test_missing_mmproj(self):
        self.helper()
        self.mmproj.unlink()
        with self.assertRaisesRegex(VisionUnavailable, "mmproj file is missing"):
            self.encode()
        self.assertIsNone(self.client.process)
        self.mmproj.write_bytes(b"fixture")
        self.assertTrue(self.encode().is_file())

    def test_missing_model(self):
        self.helper()
        self.model.unlink()
        with self.assertRaisesRegex(VisionUnavailable, "model file is missing"):
            self.encode()
        self.assertIsNone(self.client.process)

    def test_startup_failures(self):
        for mode in ("startup_exit", "startup_timeout", "startup_partial", "startup_invalid"):
            with self.subTest(mode=mode):
                self.helper(mode)
                self.assert_unavailable_and_recover()

    def test_reply_failures(self):
        for mode in ("killed", "reply_timeout", "reply_partial", "reply_invalid", "reply_error"):
            with self.subTest(mode=mode):
                self.helper(mode)
                self.assert_unavailable_and_recover()

    def test_stdin_backpressure_is_bounded(self):
        self.helper("write_timeout")
        self.assert_unavailable_and_recover(lambda: self.encode("x" * 2_000_000))

    def test_external_kill_mid_request(self):
        self.helper("reply_timeout")
        self.client.start()
        process = self.client.process
        timer = threading.Timer(0.03, lambda: os.kill(process.pid, signal.SIGKILL))
        timer.start()
        try:
            self.assert_unavailable_and_recover()
        finally:
            timer.join()

    def test_stderr_is_a_file_not_a_blocking_pipe(self):
        self.helper("stderr")
        # Writing 2 MB to stderr fills an unread pipe. It must not block READY.
        self.assertTrue(self.encode().is_file())
        self.assertGreater((self.root / "outputs/encoder.log").stat().st_size, 1_000_000)

    def test_cancel_during_encoder_response(self):
        self.helper("reply_timeout")
        cancel = threading.Event()
        timer = threading.Timer(0.03, cancel.set)
        timer.start()
        try:
            with self.assertRaisesRegex(VisionUnavailable, "cancelled"):
                self.client.encode("prompt", [self.media], cancel=cancel)
        finally:
            timer.join()
        self.assertIsNone(self.client.process)

    def test_warm_worker_and_environment(self):
        self.helper()
        self.client.env["CUDA_VISIBLE_DEVICES"] = "4"
        self.assertTrue(self.encode().is_file())
        pid = self.client.process.pid
        self.assertTrue(self.client.last_stats["worker_started"])
        self.assertTrue(self.encode().is_file())
        self.assertFalse(self.client.last_stats["worker_started"])
        self.assertEqual(pid, self.client.process.pid)

    def test_finite_deadlines_required(self):
        for key in ("timeout", "startup_timeout"):
            for value in (0, -1, float("nan"), float("inf"), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    VisionEncoder(self.adapter, self.model, self.mmproj, self.root, **{key: value})


if __name__ == "__main__":
    unittest.main()
