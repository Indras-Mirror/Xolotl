"""HAVE admission wait and native reply have separate deadlines; no GPU needed."""
import threading
import time
import unittest

from serve.server import ClientGone, ModelBusy, StrataEngine


class HaveAdmission(unittest.TestCase):
    def engine(self, batch=4):
        engine = object.__new__(StrataEngine)
        engine.batch = batch
        engine.ctl = threading.Lock()
        engine.proc = object()
        engine.gen = 7
        engine.HAVE_ADMISSION_WAIT_S = 0.5
        engine.calls = []
        def reply(keys, timeout):
            engine.calls.append((keys, timeout))
            return [(1, 1, 1)]
        engine._have_idle = reply
        return engine

    def run_have(self, engine, cancel=None):
        results, errors = [], []
        def request():
            try:
                results.append(engine.have(["frame"], timeout=0.02, cancel=cancel))
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=request)
        thread.start()
        return thread, results, errors

    def test_reply_timeout_starts_after_admission_wait(self):
        engine = self.engine()
        engine.ctl.acquire()
        thread, results, errors = self.run_have(engine)
        time.sleep(0.08)                 # admission takes longer than the native reply timeout
        self.assertTrue(thread.is_alive())
        self.assertEqual(engine.calls, [])
        engine.ctl.release()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [[(1, 1, 1)]])
        self.assertEqual(engine.calls, [(["frame"], 0.02)])

    def test_cancel_while_queued_never_sends_have(self):
        engine = self.engine()
        engine.ctl.acquire()
        cancel = threading.Event()
        thread, results, errors = self.run_have(engine, cancel)
        cancel.set()
        thread.join(2)
        engine.ctl.release()
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ClientGone)
        self.assertEqual(engine.calls, [])

    def test_admission_wait_still_has_its_own_bound(self):
        engine = self.engine()
        engine.HAVE_ADMISSION_WAIT_S = 0.03
        engine.ctl.acquire()
        thread, results, errors = self.run_have(engine)
        thread.join(2)
        engine.ctl.release()
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ModelBusy)
        self.assertEqual(engine.calls, [])

    def test_nonbatch_does_not_wait_for_batch_control(self):
        engine = self.engine(batch=0)
        engine.ctl.acquire()
        thread, results, errors = self.run_have(engine)
        thread.join(2)
        engine.ctl.release()
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [[(1, 1, 1)]])


if __name__ == "__main__":
    unittest.main()
