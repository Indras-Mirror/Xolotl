"""Cancel a real slot-condition waiter without sending STOP to another request.

Uses real StrataEngine admission and watcher methods, with no model or GPU.
"""
import io
import threading
import time
import types
import unittest

from serve import server


class ObservedCondition(threading.Condition):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.wait_count = 0
        self.notify_count = 0
        self.raise_wait = False

    def wait(self, timeout=None):
        self.wait_count += 1
        self.entered.set()
        if self.raise_wait:
            raise RuntimeError("injected wait exception")
        return super().wait(timeout)

    def notify_all(self):
        self.notify_count += 1
        return super().notify_all()


class ForbiddenCondition:
    def __enter__(self):
        raise AssertionError("active/stale STOP must not acquire slot_cv")

    def __exit__(self, *args):
        pass


class SlotAdmissionCancel(unittest.TestCase):
    def engine(self):
        engine = server.StrataEngine.__new__(server.StrataEngine)
        engine.gen = 1
        engine.info = {"batch_pinned": 1}
        engine.slot_cv = ObservedCondition()
        engine.ctl = threading.Lock()
        engine.wlock = threading.Lock()
        engine.slot_busy = [True, True]
        engine.waiting = engine.admitting = 0
        engine.sleeping = False
        engine.alive = lambda: True
        engine.can_stop = True
        engine.proc = types.SimpleNamespace(stdin=io.StringIO())
        engine.sampling_keys = lambda request: ""
        engine.pick_slot = lambda prompt: None
        engine._reap_orphan_slots = lambda: None

        def take(cancel, length, **kwargs):
            if False:
                yield None
            engine.ctl.acquire()
            return True

        engine._take_control = take
        return engine

    def worker(self, engine, cancel):
        result = {}
        done = threading.Event()

        def run():
            try:
                result["out"] = list(engine.generate_batched([1, 2], 32, {}, cancel))
            except BaseException as error:
                result["error"] = error
            finally:
                done.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def cleanup():
            cancel.set()
            engine.stop_if_running(cancel)
            thread.join(2)
            self.assertFalse(thread.is_alive())

        self.addCleanup(cleanup)
        return done, result

    def test_waiter_wakes_before_poll_timeout(self):
        engine, cancel = self.engine(), threading.Event()
        done, result = self.worker(engine, cancel)
        self.assertTrue(engine.slot_cv.entered.wait(1))
        cancel.set()
        engine.stop_if_running(cancel)
        self.assertTrue(done.wait(.3), "cancelled waiter slept until the one-second poll")
        self.assertNotIn("error", result)
        self.assertEqual(engine.admitting, 0)
        self.assertNotIn(cancel, engine._slot_admission_waiters)
        self.assertEqual(engine.proc.stdin.getvalue(), "")

    def test_cancel_before_wait_predicate_does_not_lose_wakeup(self):
        engine, cancel = self.engine(), threading.Event()
        hook = threading.Event()
        calls = 0

        def pick(prompt):
            nonlocal calls
            calls += 1
            if calls == 2:
                hook.set()
                self.assertTrue(cancel.wait(1))
            return None

        engine.pick_slot = pick
        done, result = self.worker(engine, cancel)
        self.assertTrue(hook.wait(1))
        cancel.set()
        engine.stop_if_running(cancel)
        self.assertTrue(done.wait(.3))
        self.assertNotIn("error", result)
        self.assertEqual(engine.slot_cv.wait_count, 0)
        self.assertEqual(engine.admitting, 0)

    def test_peer_notification_does_not_admit_or_cancel_peer(self):
        engine = self.engine()
        first, second = threading.Event(), threading.Event()
        done_first, result_first = self.worker(engine, first)
        done_second, result_second = self.worker(engine, second)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with engine.slot_cv:
                if len(getattr(engine, "_slot_admission_waiters", ())) == 2:
                    break
            time.sleep(.001)
        with engine.slot_cv:
            self.assertEqual(len(engine._slot_admission_waiters), 2)
        first.set()
        engine.stop_if_running(first)
        self.assertTrue(done_first.wait(.3))
        self.assertFalse(done_second.wait(.03))
        self.assertFalse(second.is_set())
        self.assertNotIn("error", result_first)
        self.assertNotIn("error", result_second)
        self.assertEqual(engine.admitting, 1)
        self.assertEqual(engine.slot_busy, [True, True])
        self.assertEqual(engine.proc.stdin.getvalue(), "")
        with engine.slot_cv:
            self.assertNotIn(first, engine._slot_admission_waiters)
            self.assertIn(second, engine._slot_admission_waiters)

    def test_wait_failure_cleans_membership_and_admission(self):
        engine, cancel = self.engine(), threading.Event()
        engine.slot_cv.raise_wait = True
        done, result = self.worker(engine, cancel)
        self.assertTrue(done.wait(1))
        self.assertIn("injected wait exception", str(result.get("error")))
        self.assertEqual(engine.admitting, 0)
        self.assertNotIn(cancel, engine._slot_admission_waiters)

    def test_native_owner_and_stale_owner_do_not_take_condition(self):
        for stale in (False, True):
            with self.subTest(stale=stale):
                engine, cancel = self.engine(), threading.Event()
                engine.slot_cv = ForbiddenCondition()
                cancel.set()
                token = object()
                engine._slot_cancel_owners = {cancel: (0, 1, engine.proc, token)}
                engine._slot_cancel_tokens = {0: object() if stale else token}
                engine.stop_if_running(cancel)
                self.assertEqual(engine.proc.stdin.getvalue(), "" if stale else "BSTOP 0\n")

    def test_solo_stop_does_not_take_condition(self):
        engine, cancel = self.engine(), threading.Event()
        engine.slot_cv = ForbiddenCondition()
        engine._running_cancel = cancel
        cancel.set()
        engine.stop_if_running(cancel)
        self.assertEqual(engine.proc.stdin.getvalue(), "STOP\n")

    def test_unrelated_event_does_not_notify(self):
        engine, cancel, peer = self.engine(), threading.Event(), threading.Event()
        engine._slot_admission_waiters = {peer}
        cancel.set()
        engine.stop_if_running(cancel)
        self.assertEqual(engine.slot_cv.notify_count, 0)
        self.assertFalse(peer.is_set())
        self.assertEqual(engine.proc.stdin.getvalue(), "")

    def test_queue_deadline_still_expires(self):
        engine, cancel = self.engine(), threading.Event()
        cancel._strata_queue_deadline = time.monotonic() + .03
        done, result = self.worker(engine, cancel)
        self.assertTrue(done.wait(.3))
        self.assertIsInstance(result.get("error"), server.QueueWaitExpired)
        self.assertEqual(engine.admitting, 0)
        self.assertNotIn(cancel, engine._slot_admission_waiters)
        self.assertEqual(engine.proc.stdin.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
