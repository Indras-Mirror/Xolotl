"""POST /sleep and /wake for any engine (serve/deep_sleep.py, docs/DEEP_SLEEP.md), without a GPU: the native pipe
protocol and the CUDA driver's checkpoint against a fake engine process and a fake driver, and the Service's part -
the drain (one request at a time and batch slots), the vision helper, a picture or a VRAM reserve while asleep.

    python -m serve.test_deep_sleep
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as S
from serve import deep_sleep as D

ENGINE = '''
import sys
from pathlib import Path
here = Path(__file__)
print("INFO prefix_cache=1 sleep=%d checkpoint_clock=%d", flush=True)
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    with here.with_suffix(".log").open("a") as log:
        log.write(line.split()[0] + "\\n")
    if line.startswith("QUIT"):
        break
    if line.startswith("SLEEP"):
        print("SLEPT seconds=0.5 kv_tokens=3", flush=True)
    elif line.startswith("WAKE"):
        if here.with_suffix(".refuse").exists():
            print("ERR wake failed (still asleep): out of memory", flush=True)
        else:
            print("INFO woke=1 wake_seconds=1.25", flush=True)
            print("READY 4096 stop", flush=True)
    elif line.startswith("CHECKPOINT_PAUSE"):
        print("CHECKPOINT_PAUSED", flush=True)
    elif line.startswith("CHECKPOINT_RESUME"):
        if here.with_suffix(".resume_ack").exists():
            print(here.with_suffix(".resume_ack").read_text(), flush=True)
        elif here.with_suffix(".refuse").exists():
            print("ERR checkpoint resume refused", flush=True)
        else:
            print("CHECKPOINT_RESUMED graph_execs=7 paused_us=1250000", flush=True)
    elif line.startswith("GEN"):
        print("T 111\\nT 107\\nDONE 2 1 1 1 stop 0 0 0 0 0", flush=True)
'''


class FakeDriver:
    """cuCheckpointProcess* on a dict: a process whose command line names `match` is a running CUDA process."""
    def __init__(self, match):
        self.match, self.states, self.calls, self.fail = match, {}, [], set()

    def state(self, pid):
        if pid in self.states:
            return self.states[pid]
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        except OSError:
            return None
        return D.RUNNING if self.match in cmd else None

    def _go(self, what, pid, state):
        self.calls.append((what, pid))
        if what in self.fail:
            raise D.CheckpointError(f"{what} of process {pid}: CUDA_ERROR_OUT_OF_MEMORY (2)")
        self.states[pid] = state

    def lock(self, pid, timeout_ms):
        self._go("lock", pid, D.LOCKED)

    def checkpoint(self, pid):
        self._go("checkpoint", pid, D.CHECKPOINTED)

    def restore(self, pid):
        self._go("restore", pid, D.LOCKED)

    def unlock(self, pid):
        self._go("unlock", pid, D.RUNNING)


class ProcessIdentity(unittest.TestCase):
    def test_reused_pid_is_never_signalled_or_restored(self):
        with mock.patch.object(D, "_process_start", return_value=10):
            pid = D.ProcessId(12345)
        driver = mock.Mock()
        with mock.patch.object(D, "_process_start", return_value=11), \
                mock.patch.object(D.os, "kill") as kill, \
                mock.patch.object(D.os, "pidfd_open", create=True) as open_pidfd:
            D._kill_processes([pid])
            with self.assertRaisesRegex(D._Stuck, "no longer matches"):
                D._restore(driver, [pid])
        kill.assert_not_called()
        open_pidfd.assert_not_called()
        driver.restore.assert_not_called()

    def test_reuse_between_identity_check_and_pidfd_open_is_not_signalled(self):
        with mock.patch.object(D, "_process_start", side_effect=[10, 10, 11]), \
                mock.patch.object(D.os, "pidfd_open", return_value=987, create=True), \
                mock.patch.object(D.signal, "pidfd_send_signal", create=True) as send, \
                mock.patch.object(D.os, "close") as close:
            pid = D.ProcessId(12345)
            D._kill_processes([pid])
        send.assert_not_called()
        close.assert_called_once_with(987)

    def test_untracked_pid_is_not_a_cleanup_target(self):
        with mock.patch.object(D.os, "kill") as kill:
            D._kill_processes([12345])
        kill.assert_not_called()

    def test_owned_process_is_killed_and_pidfd_does_not_leak(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
        try:
            tracked = D.ProcessId(proc.pid)
            self.assertTrue(tracked.matches())
            D._kill_processes([tracked])
            self.assertEqual(proc.wait(timeout=5), -9)
            self.assertFalse(tracked.matches())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


class EngineSleep(unittest.TestCase):
    def engine(self, native=False, launcher=False, mode="auto", driver=True, checkpoint_clock=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / "fake_engine.py"
        script.write_text("#!" + sys.executable + "\n" + ENGINE % (int(native), int(checkpoint_clock)), encoding="utf-8")
        script.chmod(0o700)
        exe = script
        if launcher:                                  # a launcher that does not exec: the engine is its child
            exe = Path(tmp.name) / "launch.sh"
            exe.write_text(f'#!/bin/sh\n"{script}" "$@"\n', encoding="utf-8")
            exe.chmod(0o700)
        self.drv = FakeDriver(str(script)) if driver else None
        patch = mock.patch.object(D, "driver", lambda: self.drv)
        patch.start()
        self.addCleanup(patch.stop)
        engine = S.StrataEngine(str(exe), [])
        engine.deep_sleep = mode
        self.addCleanup(engine.close)
        return engine, script

    def sent(self, script):
        log = script.with_suffix(".log")
        return log.read_text().split() if log.exists() else []

    def test_native_sleep_then_a_request_wakes_first(self):
        engine, script = self.engine(native=True)
        self.assertTrue(engine.can_sleep())
        figures = engine.sleep()
        self.assertEqual((figures["method"], figures["kv_tokens"]), ("native", 3))
        self.assertTrue(engine.sleeping)
        self.assertEqual(list(engine.generate([1], 4, {}, threading.Event())), [111, 107])
        self.assertFalse(engine.sleeping)
        self.assertEqual(engine.info["wake"]["wake_seconds"], 1.25)
        self.assertEqual(self.sent(script), ["SLEEP", "WAKE", "GEN"])
        self.assertEqual(self.drv.calls, [])             # native wins over the driver

    def test_native_wake_refused_is_gpu_busy_and_stays_asleep(self):
        engine, script = self.engine(native=True)
        engine.sleep()
        script.with_suffix(".refuse").touch()
        with self.assertRaises(S.GpuBusy):
            engine.wake()
        self.assertTrue(engine.sleeping)
        script.with_suffix(".refuse").unlink()
        engine.wake()
        self.assertFalse(engine.sleeping)

    def test_checkpoint_sleep_then_a_request_restores_first(self):
        engine, script = self.engine()
        self.assertTrue(engine.can_sleep())
        figures = engine.sleep()
        pid = engine.proc.pid
        self.assertEqual((figures["method"], figures["processes"]), ("checkpoint", 1))
        self.assertEqual(self.drv.states[pid], D.CHECKPOINTED)
        self.assertEqual(self.drv.calls, [("lock", pid), ("checkpoint", pid)])
        self.assertEqual(engine.info["asleep"]["method"], "checkpoint")
        self.assertEqual(list(engine.generate([1], 4, {}, threading.Event())), [111, 107])
        self.assertEqual(self.drv.calls[2:], [("restore", pid), ("unlock", pid)])
        self.assertEqual(self.drv.states[pid], D.RUNNING)
        self.assertEqual(self.sent(script), ["GEN"])     # the engine itself saw nothing of it
        self.assertIn("restore_seconds", engine.info["wake"])

    def test_checkpoint_hook_runs_after_unlock_before_generation(self):
        engine, script = self.engine(checkpoint_clock=True)
        engine.sleep()
        command = D._command
        def after_unlock(*args, **kwargs):
            self.assertEqual(self.drv.states[engine.proc.pid], D.RUNNING)
            self.assertEqual(self.drv.calls[-1], ("unlock", engine.proc.pid))
            return command(*args, **kwargs)
        with mock.patch.object(D, "_command", side_effect=after_unlock):
            self.assertEqual(list(engine.generate([1], 4, {}, threading.Event())), [111, 107])
        self.assertEqual(self.sent(script), ["CHECKPOINT_PAUSE", "CHECKPOINT_RESUME", "GEN"])
        self.assertEqual(engine.info["wake"]["graph_execs"], 7)
        self.assertEqual(engine.info["wake"]["paused_us"], 1250000)

    def test_checkpoint_hook_failure_ends_restored_processes(self):
        for failure in ("refusal", "timeout"):
            with self.subTest(failure=failure):
                engine, script = self.engine(checkpoint_clock=True)
                engine.sleep()
                script.with_suffix(".refuse").touch()
                command = D._command
                if failure == "timeout":
                    command = mock.Mock(side_effect=D.EngineGone("reset timed out"))
                with mock.patch.object(D, "_command", side_effect=command):
                    with self.assertRaisesRegex(S.EngineDied, "checkpoint-controlled engine was ended"):
                        engine.wake()
                engine.proc.wait(timeout=5)
                self.assertFalse(engine.alive())
                self.assertFalse(engine.sleeping)
                self.assertNotIn("asleep", engine.info)
                self.assertNotIn("GEN", self.sent(script))

    def assert_bad_checkpoint_ack_ends_engine(self, acknowledgement):
        engine, script = self.engine(checkpoint_clock=True)
        engine.sleep()
        tracked = list(engine.sleep_pids)
        script.with_suffix(".resume_ack").write_text(acknowledgement)
        with mock.patch.object(D, "_kill_processes", wraps=D._kill_processes) as cleanup:
            with self.assertRaisesRegex(S.EngineDied, "checkpoint-controlled engine was ended"):
                engine.wake()
            cleanup.assert_called_once_with(tracked)
        engine.proc.wait(timeout=5)
        self.assertFalse(engine.alive())
        self.assertFalse(engine.sleeping)
        self.assertFalse(engine.checkpoint_paused)
        self.assertNotIn("asleep", engine.info)
        self.assertNotIn("GEN", self.sent(script))

    def test_checkpoint_ack_requires_exact_token(self):
        self.assert_bad_checkpoint_ack_ends_engine(
            "CHECKPOINT_RESUMED_INVALID paused_us=1 graph_execs=7")

    def test_checkpoint_ack_requires_both_fields_on_ack(self):
        for ack in ("CHECKPOINT_RESUMED", "CHECKPOINT_RESUMED paused_us=1",
                    "INFO paused_us=1 graph_execs=7\nCHECKPOINT_RESUMED"):
            with self.subTest(ack=ack):
                self.assert_bad_checkpoint_ack_ends_engine(ack)

    def test_checkpoint_ack_rejects_invalid_telemetry(self):
        for key in ("paused_us", "graph_execs"):
            for value in ("-1", "1.0", "nan", "bad"):
                fields = {"paused_us": "1", "graph_execs": "7", key: value}
                with self.subTest(key=key, value=value):
                    self.assert_bad_checkpoint_ack_ends_engine(
                        "CHECKPOINT_RESUMED " + " ".join(f"{k}={v}" for k, v in fields.items()))

    def test_checkpoint_hook_is_not_sent_until_restore_succeeds(self):
        engine, script = self.engine(checkpoint_clock=True)
        engine.sleep()
        self.drv.fail.add("restore")
        with self.assertRaises(S.GpuBusy):
            engine.wake()
        self.assertEqual(self.sent(script), ["CHECKPOINT_PAUSE"])
        self.assertTrue(engine.sleeping)
        self.assertTrue(engine.checkpoint_paused)
        self.drv.fail.clear()
        engine.wake()
        self.assertEqual(self.sent(script), ["CHECKPOINT_PAUSE", "CHECKPOINT_RESUME"])
        self.assertFalse(engine.checkpoint_paused)

    def test_checkpoint_failure_resumes_clock_after_rollback(self):
        engine, script = self.engine(checkpoint_clock=True)
        self.drv.fail.add("checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint failed"):
            engine.sleep()
        self.assertEqual(self.sent(script), ["CHECKPOINT_PAUSE", "CHECKPOINT_RESUME"])
        self.assertEqual(self.drv.states[engine.proc.pid], D.RUNNING)
        self.assertFalse(engine.sleeping)
        self.assertFalse(engine.checkpoint_paused)

    def test_checkpoint_clock_does_not_survive_process_restart(self):
        engine, _ = self.engine(checkpoint_clock=True)
        engine.sleep()
        self.assertTrue(engine.checkpoint_paused)
        engine.unload()
        engine.restart()
        self.assertFalse(engine.sleeping)
        self.assertFalse(engine.checkpoint_paused)
        self.assertEqual(engine.sleep_pids, [])

    def test_pause_failure_does_not_checkpoint_or_admit_work(self):
        engine, script = self.engine(checkpoint_clock=True)
        with mock.patch.object(D, "_command", return_value=(None, "pause refused", {})):
            with self.assertRaisesRegex(S.EngineDied, "checkpoint-controlled engine was ended"):
                engine.sleep()
        engine.proc.wait(timeout=5)
        self.assertFalse(engine.alive())
        self.assertEqual(self.drv.calls, [])
        self.assertNotIn("GEN", self.sent(script))

    def test_checkpoint_restore_refused_stays_asleep(self):
        engine, _ = self.engine()
        engine.sleep()
        self.drv.fail.add("restore")
        with self.assertRaisesRegex(S.GpuBusy, "OUT_OF_MEMORY"):
            engine.wake()
        self.assertTrue(engine.sleeping)
        self.assertEqual(self.drv.states[engine.proc.pid], D.CHECKPOINTED)
        self.drv.fail.clear()
        engine.wake()
        self.assertFalse(engine.sleeping)

    def test_unlock_failure_ends_engine_instead_of_retrying_restore(self):
        engine, _ = self.engine()
        engine.sleep()
        self.drv.fail.add("unlock")
        with self.assertRaisesRegex(S.EngineDied, "could not all be unlocked"):
            engine.wake()
        engine.proc.wait(timeout=5)
        self.assertFalse(engine.alive())

    def test_closing_a_checkpointed_launcher_ends_cuda_child(self):
        engine, _ = self.engine(launcher=True)
        engine.sleep()
        (pid,) = engine.sleep_pids
        with mock.patch.object(D, "_kill_processes", wraps=D._kill_processes) as kill:
            engine.unload()
        kill.assert_any_call([pid])
        # A zombie is already terminated; its launcher may have been killed before reaping.
        status = Path(f"/proc/{pid}/status")
        if status.exists():
            self.assertIn("State:\tZ", status.read_text())

    def test_checkpoint_refused_unlocks_and_stays_awake(self):
        engine, _ = self.engine()
        self.drv.fail.add("checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint failed"):
            engine.sleep()
        self.assertFalse(engine.sleeping)
        self.assertEqual(self.drv.states[engine.proc.pid], D.RUNNING)

    def test_the_launchers_child_is_checkpointed(self):
        engine, _ = self.engine(launcher=True)
        engine.sleep()
        (pid,) = engine.sleep_pids
        self.assertNotEqual(pid, engine.proc.pid)
        self.assertIn("fake_engine", Path(f"/proc/{pid}/cmdline").read_text(errors="replace"))
        engine.wake()

    def test_closing_a_checkpointed_engine_kills_it(self):
        engine, script = self.engine()
        engine.sleep()
        proc = engine.proc
        engine.close()
        self.assertEqual(proc.returncode, -9)
        self.assertNotIn("QUIT", self.sent(script))      # QUIT would wait on CUDA calls that cannot run

    def test_modes(self):
        engine, _ = self.engine(native=True, mode="checkpoint")
        self.assertEqual(engine.sleep()["method"], "checkpoint")
        engine.wake()
        engine.deep_sleep = "off"
        self.assertFalse(engine.can_sleep())
        with self.assertRaisesRegex(ValueError, "cannot sleep"):
            engine.sleep()

    def test_no_native_sleep_and_no_driver(self):
        engine, script = self.engine(driver=False)
        self.assertFalse(engine.can_sleep())
        with self.assertRaisesRegex(ValueError, "cannot sleep"):
            engine.sleep()
        self.assertEqual(self.sent(script), [])

    def test_default_off_does_not_probe_the_driver_or_send_commands(self):
        engine, script = self.engine(native=True, mode="off")
        del engine.deep_sleep                 # exercise the engine class default
        with mock.patch.object(D, "driver") as driver:
            self.assertFalse(engine.can_sleep())
            with self.assertRaisesRegex(ValueError, "cannot sleep"):
                engine.sleep()
        driver.assert_not_called()
        self.assertEqual(self.sent(script), [])

    def test_mode_from_config(self):
        self.assertEqual(D.mode_from_config({}), "off")
        self.assertEqual(D.mode_from_config({"deep_sleep": None}), "off")
        self.assertEqual(D.mode_from_config({"deep_sleep": True}), "auto")
        self.assertEqual(D.mode_from_config({"deep_sleep": False}), "off")
        self.assertEqual(D.mode_from_config({"deep_sleep": "checkpoint"}), "checkpoint")
        with self.assertRaises(ValueError):
            D.mode_from_config({"deep_sleep": "always"})


class FakeEngine(S.MockEngine):
    """An engine object for the Service: sleeps and wakes on flags, records VRAM commands."""
    sleeping = False

    def __init__(self, *a, batch=0, **k):
        super().__init__(*a, **k)
        self.info, self.vrams, self.events = {}, [], []
        self.batch = batch
        if batch:
            self.ctl, self.slot_busy = threading.Lock(), [False] * batch

    def alive(self):
        return True

    def can_sleep(self):
        return True

    def sleep(self):
        self.events.append("sleep")
        self.sleeping = True
        return {"method": "checkpoint", "seconds": 0.5}

    def wake(self):
        self.events.append("wake")
        self.sleeping = False
        self.info["wake"] = {"wake_seconds": 1.0}
        return self.info["wake"]

    def unload(self):
        self.events.append("unload")
        self.sleeping = False

    def vram(self, reserve):
        assert not self.sleeping, "VRAM reached a sleeping engine"
        self.vrams.append(reserve)
        return {}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        assert not self.sleeping, "a request reached a sleeping engine"
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class FakeVision:
    lazy_vision, elastic = True, False

    def __init__(self, events):
        self.running, self.events = True, events

    def alive(self):
        return self.running

    def unload(self):
        self.events.append("vision down")
        self.running = False

    close = unload


class ServiceSleep(unittest.TestCase):
    def service(self, **k):
        engine = FakeEngine(S.ByteTokenizer(), "ok", max_context=256, **k)
        svc = S.Service(engine, S.ByteTokenizer(), S.ChatTemplate(S.ROOT / "serve/chat_template.jinja"))
        svc.free_vram_mib = lambda: 32000
        return svc, engine

    def test_sleep_figures_and_states(self):
        svc, engine = self.service()
        self.assertEqual(svc.sleep(), {"status": "asleep", "method": "checkpoint", "seconds": 0.5,
                                       "vram_free_mib": 32000})
        self.assertTrue(svc._asleep())
        self.assertEqual(svc.sleep()["status"], "asleep")   # once
        self.assertEqual(engine.events, ["sleep"])
        self.assertEqual(svc.wake()["status"], "awake")
        self.assertEqual(svc.wake(), {"status": "not asleep"})

    def test_unsupported_and_busy(self):
        svc, engine = self.service()
        engine.can_sleep = lambda: False
        self.assertEqual(svc.sleep(), {"status": "unsupported"})
        engine.can_sleep = lambda: True
        with svc.status_lock:
            svc.status["busy"] = True
        self.assertEqual(svc.sleep(), {"status": "busy"})
        self.assertEqual(engine.events, [])

    def test_batch_slots_drain_first(self):
        svc, engine = self.service(batch=4)
        engine.slot_busy[2] = True
        self.assertEqual(svc.sleep(), {"status": "busy"})
        engine.slot_busy[2] = False
        with engine.ctl:                                   # a prompt being admitted
            self.assertEqual(svc.sleep(), {"status": "busy"})
        self.assertEqual(svc.sleep()["status"], "asleep")
        self.assertFalse(engine.ctl.locked())

    def test_the_vision_helper_stops_and_a_picture_wakes_the_engine_first(self):
        svc, engine = self.service()
        svc.vision = FakeVision(engine.events)
        self.assertEqual(svc.sleep()["vision"], "stopped")
        self.assertEqual(engine.events, ["vision down", "sleep"])
        with svc._turn():                                 # an image encode's turn
            self.assertFalse(engine.sleeping)
        self.assertEqual(engine.events[-1], "wake")

    def test_a_cuda_helper_is_checkpointed_with_the_engine(self):
        svc, engine = self.service()
        helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)  # fake-helper"])
        self.addCleanup(helper.kill)
        while b"fake-helper" not in Path(f"/proc/{helper.pid}/cmdline").read_bytes():   # until it has exec'd
            time.sleep(0.01)
        svc.vision = FakeVision(engine.events)
        svc.vision.proc = helper
        drv = FakeDriver("fake-helper")
        with mock.patch.object(D, "driver", lambda: drv):
            self.assertEqual(svc.sleep()["vision"], "checkpointed")
            self.assertEqual(engine.events, ["sleep"])          # not stopped: it keeps its state and its VRAM size
            self.assertEqual(drv.states[helper.pid], D.CHECKPOINTED)
            svc.wake()
            self.assertEqual(drv.states[helper.pid], D.RUNNING)
            self.assertEqual(engine.events, ["sleep", "wake"])
            engine.sleep = lambda: (_ for _ in ()).throw(ValueError("refused"))
            with self.assertRaises(ValueError):
                svc.sleep()
            self.assertEqual(drv.states[helper.pid], D.RUNNING)  # a failed sleep restores the helper
            del engine.sleep
            svc.sleep()
            svc.unload()                                         # a checkpointed helper is ended, not asked to quit
        self.assertEqual(helper.wait(timeout=10), -9)

    def frozen_helper(self, svc, engine):
        svc.vision = FakeVision(engine.events)
        svc.vision_frozen = [123456789]
        engine.sleeping = True

    def test_dead_engine_unload_ends_frozen_helper(self):
        svc, engine = self.service()
        self.frozen_helper(svc, engine)
        engine.alive = lambda: False
        with mock.patch.object(D, "_kill_processes") as kill:
            self.assertEqual(svc.unload(), "not loaded")
        kill.assert_called_once_with([123456789])
        self.assertIsNone(svc.vision_frozen)
        self.assertFalse(svc.vision.alive())

    def test_dead_engine_restart_ends_frozen_helper_first(self):
        svc, engine = self.service()
        self.frozen_helper(svc, engine)
        engine.alive = lambda: False
        def restart():
            self.assertIsNone(svc.vision_frozen)
            self.assertFalse(svc.vision.alive())
            engine.sleeping = False
            engine.alive = lambda: True
            engine.events.append("restart")
        engine.restart = restart
        with mock.patch.object(D, "_kill_processes") as kill:
            svc.load()
        kill.assert_called_once_with([123456789])
        self.assertEqual(engine.events, ["vision down", "restart"])

    def test_fatal_wake_ends_helper_but_busy_wake_preserves_it(self):
        for error in (S.EngineDied("partial unlock"), S.GpuBusy("restore needs VRAM")):
            with self.subTest(error=type(error).__name__):
                svc, engine = self.service()
                self.frozen_helper(svc, engine)
                with mock.patch.object(engine, "wake", side_effect=error), mock.patch.object(D, "_kill_processes") as kill:
                    with self.assertRaises(type(error)):
                        svc.wake()
                if isinstance(error, S.EngineDied):
                    kill.assert_called_once_with([123456789])
                    self.assertIsNone(svc.vision_frozen)
                    self.assertFalse(svc.vision.alive())
                else:
                    kill.assert_not_called()
                    self.assertEqual(svc.vision_frozen, [123456789])
                    self.assertTrue(svc.vision.alive())
                    self.assertTrue(engine.sleeping)

    def test_elastic_idle_rechecks_sleep_after_taking_fifo(self):
        svc, engine = self.service()
        svc.vision = FakeVision(engine.events)
        svc.vision.elastic = True
        svc.vision.elastic_idle_s = 10
        svc.last_image_at = 0
        lock = mock.Mock()
        def acquire(**kwargs):
            # The concurrent sleep won the turn immediately before this acquire.
            engine.sleeping = True
            svc.vision_frozen = [123456789]
            return True
        lock.acquire.side_effect = acquire
        svc.fifo = lock
        self.assertEqual(svc._vision_idle_down(), "busy")
        lock.release.assert_called_once_with()
        self.assertTrue(svc.vision.alive())
        self.assertEqual(svc.vision_frozen, [123456789])
        self.assertEqual(engine.events, [])

    def test_an_elastic_cache_takes_the_encoders_vram_back_before_sleep(self):
        svc, engine = self.service()
        svc.vision = FakeVision(engine.events)
        svc.vision.lazy_vision = False
        svc.vision_held = True
        svc.sleep()
        self.assertEqual(engine.vrams, [0])
        self.assertFalse(svc.vision_held)

    def test_a_vram_reserve_asked_while_asleep_applies_at_the_wake(self):
        svc, engine = self.service()
        svc.sleep()
        self.assertEqual(svc.vram(1500)["status"], "asleep")
        self.assertEqual(engine.vrams, [])
        svc.wake()
        self.assertEqual(engine.vrams, [1500])

    def test_busy_gpu_refuses_wake_and_preserves_sleep(self):
        svc, engine = self.service()
        svc.sleep()
        svc.min_free_vram_mib = 30000
        svc.vram_free_wait_s = 0
        svc.free_vram_mib = lambda: 20000
        with self.assertRaises(S.GpuBusy):
            svc.wake()
        self.assertTrue(engine.sleeping)
        self.assertEqual(engine.events, ["sleep"])
        svc.free_vram_mib = lambda: 32000
        self.assertEqual(svc.wake()["status"], "awake")

    def test_a_request_wakes_the_engine(self):
        svc, engine = self.service()
        svc.sleep()
        svc.load()
        self.assertEqual(engine.events, ["sleep", "wake"])


if __name__ == "__main__":
    unittest.main()
