"""A stalled done-log stdout must not retain status_lock or the engine FIFO (CPU only)."""
import io
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]


class BlockingStdout(io.StringIO):
    def __init__(self, fail=False):
        super().__init__()
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.fail = fail
        self.old_wrote_done = False

    def write(self, text):
        if threading.current_thread().name == 'done-log-old' and text.startswith('[strata] done:'):
            self.old_wrote_done = True
        return super().write(text)

    def flush(self):
        if threading.current_thread().name == 'done-log-old' and self.old_wrote_done:
            self.blocked.set()
            if self.fail:
                raise OSError('synthetic stdout failure')
            if not self.release.wait(5):
                raise TimeoutError('test did not release synthetic stdout')


class DoneLogUnlock(unittest.TestCase):
    def service(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, ['old', 'newer']), tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
        svc.api_monitor = True
        # Enable request IDs/timestamps in the done line without opening a file.
        svc.request_log = types.SimpleNamespace(detail=False)
        return svc

    def check_blocked_done(self, cancel_old):
        svc = self.service()
        stdout = BlockingStdout()
        errors, records, replies = [], {}, {}
        next_done = threading.Event()

        def run_old():
            try:
                records['old'] = svc.begin_request('/v1/chat/completions', {'messages': [], 'stream': True})
                cancel = threading.Event()
                gen = svc.run(svc.tok.encode('first'), False, None, 30, {}, cancel)
                if cancel_old:
                    next(gen)
                    cancel.set()
                    gen.close()  # GeneratorExit must release both locks before its done log blocks.
                else:
                    replies['old'] = list(gen)
            except BaseException as e:
                errors.append(e)

        def run_next():
            try:
                # Both the monitor path and request-log path acquire status_lock.
                records['next'] = svc.begin_request('/v1/chat/completions', {'messages': [], 'stream': True})
                with svc.status_lock:
                    self.assertFalse(svc.status['busy'])
                self.assertTrue(svc.fifo.acquire(timeout=.5), 'old done print retained FIFO')
                svc.fifo.release()
                replies['next'] = list(svc.run(svc.tok.encode('second'), False, None, 30, {}, threading.Event()))
            except BaseException as e:
                errors.append(e)
            finally:
                next_done.set()

        old = threading.Thread(target=run_old, name='done-log-old')
        nxt = threading.Thread(target=run_next, name='done-log-next')
        with mock.patch.object(sys, 'stdout', stdout):
            try:
                old.start()
                self.assertTrue(stdout.blocked.wait(2), 'old request never reached done stdout.flush')
                nxt.start()
                self.assertTrue(next_done.wait(1), 'next begin_request/status/FIFO remained blocked behind stdout')
                self.assertTrue(old.is_alive(), 'old stdout must still be blocked during the next request')
                self.assertEqual(errors, [])
            finally:
                stdout.release.set()
                old.join(3)
                if nxt.ident is not None:
                    nxt.join(3)
        self.assertFalse(old.is_alive())
        self.assertFalse(nxt.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(next(data['completion_tokens'] for kind, data in replies['next'] if kind == 'done'), 6)
        lines = stdout.getvalue().splitlines()
        old_line = next(line for line in lines if 'request ' + records['old']['id'] in line)
        next_line = next(line for line in lines if 'request ' + records['next']['id'] in line)
        self.assertIn('1 tokens' if cancel_old else '4 tokens', old_line)
        self.assertIn('6 tokens', next_line)
        self.assertIn('cancel=True' if cancel_old else 'cancel=False', old_line)
        self.assertNotIn(records['next']['id'], old_line)
        self.assertFalse(svc.status['busy'])
        self.assertFalse(svc.fifo.locked())

    def test_next_request_runs_while_previous_done_stdout_is_blocked(self):
        self.check_blocked_done(False)

    def test_cancelled_generator_releases_locks_before_done_stdout(self):
        self.check_blocked_done(True)

    def test_stdout_exception_cannot_retain_locks(self):
        svc = self.service()
        stdout = BlockingStdout(fail=True)
        errors = []
        def run():
            try:
                svc.begin_request('/v1/chat/completions', {'messages': []})
                list(svc.run(svc.tok.encode('first'), False, None, 30, {}, threading.Event()))
            except BaseException as e:
                errors.append(e)
        with mock.patch.object(sys, 'stdout', stdout):
            t = threading.Thread(target=run, name='done-log-old')
            t.start(); t.join(3)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], OSError)
        self.assertTrue(svc.status_lock.acquire(timeout=.5))
        try:
            self.assertFalse(svc.status['busy'])
        finally:
            svc.status_lock.release()
        self.assertTrue(svc.fifo.acquire(timeout=.5))
        svc.fifo.release()


if __name__ == '__main__':
    unittest.main()
