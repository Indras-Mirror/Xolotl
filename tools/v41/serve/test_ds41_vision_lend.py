"""#146: the keyed image path takes the turn's VRAM lend and stops a per_image encoder (CPU, no encoder process)."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_ds41 as D
from serve.test_image_ids import KeyedVision, RefEngine, chat, png
from test_ds41_image_ids import DsTokenizer


def bare_vision(per_image=False, standby=False):
    """A Ds41Vision without __init__ (no process): only the lifecycle attributes the lend / per_image code reads."""
    v = object.__new__(D.Ds41Vision)
    v.lend_cb, v.per_image, v.standby, v.cache = None, per_image, standby, {}
    v._spawned = threading.Event()
    v._spawned.set()
    v.log = []
    v._alive = False
    v.alive = lambda: v._alive
    v._start = lambda: (v.log.append("start"), setattr(v, "_alive", True))
    v.unload = lambda: (v.log.append("unload"), setattr(v, "_alive", False))
    v._respawn = lambda: v.log.append("respawn")
    return v


def fake_encode_keyed(self, key, data, keep=()):
    self.log.append("enc " + key)
    self.cache[key] = (Path("/nonexistent/" + key), 4)
    return self.cache[key]


class Ds41VisionLend(unittest.TestCase):
    def setUp(self):
        p = patch.object(D.Ds41Vision.__mro__[1], "encode_keyed", fake_encode_keyed)   # S.Vision is Ds41Vision itself
        p.start()
        self.addCleanup(p.stop)

    def test_lend_taken_once_on_first_uncached_picture(self):
        v = bare_vision()
        v.cache["old"] = (Path("/x"), 4)
        v.lend_cb = lambda: v.log.append("LEND")
        v.encode_keyed("old", b"")                 # cached: no lend
        v.encode_keyed("a", b"")
        v.encode_keyed("b", b"")
        v.turn_done()
        self.assertEqual(v.log, ["enc old", "LEND", "enc a", "enc b"])
        self.assertIsNone(v.lend_cb)

    def test_per_image_starts_then_stops_and_respawns(self):
        v = bare_vision(per_image=True, standby=True)
        v.encode_keyed("a", b"")
        v.turn_done()
        self.assertEqual(v.log, ["start", "enc a", "unload", "respawn"])

    def test_turn_done_drops_an_unused_lend(self):
        v = bare_vision()
        v.lend_cb = lambda: v.log.append("LEND")
        v.turn_done()
        self.assertEqual(v.log, [])
        self.assertIsNone(v.lend_cb)


class CountingVision(KeyedVision):
    def __init__(self, d):
        super().__init__(d)
        self.turns_done = 0

    def turn_done(self):
        self.turns_done += 1


class ServerKeyedPathCallsTurnDone(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tok = DsTokenizer()
        self.vision = CountingVision(self.tmp.name)
        self.svc = D.S.Service(RefEngine(self.tok, "ok", max_context=32768), self.tok, D.Ds41Template(),
                               vision=self.vision)

    def tearDown(self):
        self.svc.drop_embeddings()
        self.tmp.cleanup()

    def test_only_a_turn_that_encoded_ends_the_encoder(self):
        img = {"type": "image_url", "image_url": {"url": png(1)}}
        messages, tools, kwargs = chat(img)
        self.svc.prepare(messages, tools, kwargs, 8)
        self.assertEqual((self.vision.encodes, self.vision.turns_done), (1, 1))
        messages, tools, kwargs = chat(img)        # the same picture again: cached, nothing to stop
        self.svc.prepare(messages, tools, kwargs, 8)
        self.assertEqual((self.vision.encodes, self.vision.turns_done), (1, 1))


if __name__ == "__main__":
    unittest.main()
