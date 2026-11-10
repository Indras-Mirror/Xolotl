"""serve/test_engine_start_race.py: a failed engine start no longer crashes when another thread closed the process.

StrataEngine.__init__ cleans up a start that ended before READY.  restart()/close() on another thread sets self.proc to
None, so reading self.proc there raised AttributeError ("NoneType has no attribute 'stdin'") and a recoverable engine
abort reached the client as a 400.  The cleanup now uses the process object of THIS start, captured once right after it
was spawned (upstream #4792).  No GPU or model: the "engine" is a script that exits before READY.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server  # noqa: E402

#: exits at once, before READY: the start that must be cleaned up
FAILING = "import sys; sys.exit(3)\n"


class EngineStartRace(unittest.TestCase):
    def engine(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / "engine.py"
        script.write_text(FAILING, encoding="utf-8")
        holder = {}
        real_init = server.StrataEngine.__init__
        real_popen = server.subprocess.Popen

        def init_stashing_self(self, *a, **k):       # the instance exists before __init__ runs
            holder["eng"] = self
            return real_init(self, *a, **k)

        def racing_popen(cmd, **kw):
            proc = real_popen([sys.executable, str(script), *cmd[1:]], **kw)
            real_wait = proc.wait

            def wait_then_another_thread_closes(*a, **k):
                out = real_wait(*a, **k)
                holder["eng"].proc = None             # what restart()/close() does on another thread
                return out

            proc.wait = wait_then_another_thread_closes
            return proc

        with mock.patch.object(server.StrataEngine, "__init__", init_stashing_self), \
                mock.patch.object(server.subprocess, "Popen", racing_popen), \
                mock.patch.object(server, "START_RETRIES", 0):     # one attempt: the cleanup runs straight away
            eng = server.StrataEngine(sys.executable, ["--max-context", "4096"])
        self.addCleanup(eng.close)
        return eng

    def test_a_failed_start_survives_a_concurrent_close(self):
        try:
            self.engine()
        except AttributeError as e:                  # the bug: "NoneType object has no attribute 'stdin'"
            self.fail(f"a failed start raised AttributeError instead of reporting the failure cleanly: {e}")
        except RuntimeError:
            pass                                     # the start failed, and it said so: that is the fix working
        except Exception as e:                       # noqa: BLE001 - any other type is still a bug
            self.fail(f"a failed start raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
