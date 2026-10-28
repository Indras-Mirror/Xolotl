"""Model-free checks for M31 image/video -> common GENI and failure isolation."""
import base64
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools/m31/serve")]
import serve_m31 as M
from serve import frontend as F
from serve import server as S
from vision_http import MediaInput, install_video_reader


def fixture(directory):
    directory = Path(directory)
    chunks = [
        {"index": 0, "row": 0, "position": 0, "n_pos": 1, "rows": 1, "type": "text", "tokens": [41]},
        {"index": 1, "row": 1, "position": 1, "n_pos": 2, "rows": 2, "type": "image", "visual_row": 0,
         "embeddings": "frame0.f32"},
        {"index": 2, "row": 3, "position": 3, "n_pos": 1, "rows": 1, "type": "text", "tokens": [42]},
        {"index": 3, "row": 4, "position": 4, "n_pos": 1, "rows": 1, "type": "image", "visual_row": 2,
         "embeddings": "frame1.f32"},
        {"index": 4, "row": 5, "position": 5, "n_pos": 1, "rows": 1, "type": "text", "tokens": [43]},
    ]
    value = {"format": "m31-vision-ipc-v1", "source_pr": "6b95a9908d2c7b242817a447708aec38e937e2c0",
             "n_embd": 6144, "llm_inference": False, "encoded": True, "rows": 6, "visual_rows": 3,
             "chunks": chunks}
    (directory / "frame0.f32").write_bytes(struct.pack("<f", 0.25) * 12288)
    (directory / "frame1.f32").write_bytes(struct.pack("<f", -0.5) * 6144)
    path = directory / "manifest.json"
    path.write_text(json.dumps(value))
    return path, value


class MixedEmbeddingTests(unittest.TestCase):
    def test_text_and_video_frames_use_one_ordered_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = fixture(tmp)
            output = Path(tmp) / "request.sve"
            ids = M.mixed_embeddings(path, output)
            self.assertEqual(ids, [41, -1, -1, 42, -1, 43])
            expected = (struct.pack("<5i", 0x31455653, 2, 2, 1, 6144) +
                        (Path(tmp) / "frame0.f32").read_bytes() +
                        struct.pack("<5i", 0x31455653, 1, 1, 1, 6144) +
                        (Path(tmp) / "frame1.f32").read_bytes())
            self.assertEqual(output.read_bytes(), expected)

    def test_corrupt_helper_results_are_rejected_before_native_send(self):
        for change in (lambda m: m.update(source_pr="wrong"),
                       lambda m: m.update(encoded=False),
                       lambda m: m["chunks"][1].update(row=2),
                       lambda m: m["chunks"][1].update(visual_row=1),
                       lambda m: m["chunks"][1].update(embeddings="../frame0.f32"),
                       lambda m: m["chunks"][0].update(tokens=[-1]),
                       lambda m: m.update(visual_rows=4)):
            with tempfile.TemporaryDirectory() as tmp:
                path, value = fixture(tmp)
                change(value)
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "vision unavailable"):
                    M.mixed_embeddings(path, Path(tmp) / "request.sve")
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = fixture(tmp)
            with self.assertRaisesRegex(ValueError, "vision unavailable"):
                M.mixed_embeddings(path, Path(tmp) / "request.sve", max_rows=3)

    def test_embedding_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = fixture(tmp)
            embedding = Path(tmp) / "frame0.f32"
            embedding.unlink()
            embedding.symlink_to(Path(tmp) / "frame1.f32")
            with self.assertRaisesRegex(ValueError, "vision unavailable"):
                M.mixed_embeddings(path, Path(tmp) / "request.sve")


class MediaTests(unittest.TestCase):
    def test_sources_are_kept_until_last_prepared_request_releases_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = MediaInput(tmp)
            try:
                with mock.patch.object(media, "_validate"):
                    first = media._store(b"same", "image", 1)
                    second = media._store(b"same", "image", 1)
                self.assertEqual(first, second)
                media.release([first])
                self.assertTrue(second.is_file())
                media.release([second])
                self.assertFalse(second.exists())
                self.assertEqual(media.references, {})
            finally:
                media.close()

    def test_preparation_failure_releases_earlier_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = MediaInput(tmp)
            messages = [{"role": "user", "content": [{"type": "image", "source": "a"},
                                                         {"type": "image", "source": "b"}]}]
            with mock.patch.object(media, "_read", side_effect=[b"first", ValueError("bad media")]), \
                    mock.patch.object(media, "_validate"):
                with self.assertRaisesRegex(ValueError, "bad media"):
                    media.prepare(messages)
            self.assertEqual(list(media.dir.iterdir()), [])
            self.assertEqual(media.references, {})

    def test_common_normalized_image_and_video_sources_keep_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = MediaInput(tmp)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "source": "first.png"},
                {"type": "image", "source": "clip.mp4", "media_type": "video", "fps": 2},
                {"type": "text", "text": "after"}]}]
            before = copy.deepcopy(messages)
            with mock.patch.object(media, "_read", side_effect=[b"first", b"clip"]), \
                    mock.patch.object(media, "_store", side_effect=[Path("a"), Path("b")]) as store:
                copied, paths, markers, fps = media.prepare(messages)
            self.assertEqual(messages, before)
            self.assertEqual(paths, [Path("a"), Path("b")])
            self.assertEqual(fps, 2)
            self.assertEqual(store.call_args_list[0].args, (b"first", "image", 1.0))
            self.assertEqual(store.call_args_list[1].args, (b"clip", "video", 2.0))
            self.assertEqual(copied[0]["content"], "before" + markers[0] + markers[1] + "after")

    def test_video_extension_is_local_to_explicit_install_and_preserves_images(self):
        names = ("IMAGE_PARTS", "ANTHROPIC_BLOCKS", "_parts_of", "_m31_video_reader")
        before = {name: getattr(F, name, None) for name in names}
        try:
            install_video_reader()
            parts = F._parts_of([{"type": "image_url", "image_url": {"url": "image.png"}},
                                 {"type": "video_url", "video_url": {"url": "video.mp4", "fps": 1}},
                                 {"type": "text", "text": "describe"}])
            self.assertEqual(parts[0]["source"], "image.png")
            self.assertEqual(parts[1], {"type": "image", "source": "video.mp4", "media_type": "video", "fps": 1})
            self.assertEqual(parts[2], {"type": "text", "text": "describe"})
        finally:
            for name, value in before.items():
                if value is None:
                    if hasattr(F, name):
                        delattr(F, name)
                else:
                    setattr(F, name, value)

    def test_cancel_before_media_read_and_fps_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = MediaInput(tmp)
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(S.ClientGone):
                media.prepare([{"role": "user", "content": []}], cancel=cancel)
            for fps in (False, float("nan"), float("inf"), 0, 31):
                with self.assertRaisesRegex(ValueError, "video_fps"):
                    media.prepare([], fps)


class WrapperTests(unittest.TestCase):
    def test_missing_helper_is_deferred_and_environment_is_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            vision = M.M31Vision({"exe": str(Path(tmp) / "missing"), "model": "missing-model",
                                  "mmproj": "missing-mmproj", "cache_dir": str(Path(tmp) / "cache")},
                                 env={"CUDA_VISIBLE_DEVICES": "test-visible"})
            try:
                self.assertFalse(vision.alive())
                self.assertTrue(vision.lazy_vision)
                self.assertEqual(vision.client.env, {"CUDA_VISIBLE_DEVICES": "test-visible"})
                with self.assertRaisesRegex(S.VisionUnavailable, "vision unavailable: helper executable"):
                    vision.encode_prompt("<__media__>", [Path(tmp)], 1)
                self.assertFalse(vision.alive())
            finally:
                vision.shutdown()

    def test_cancelled_helper_becomes_common_client_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            vision = M.M31Vision({"exe": str(Path(tmp) / "missing"), "model": "missing-model",
                                  "mmproj": "missing-mmproj", "cache_dir": str(Path(tmp) / "cache")})
            cancel = threading.Event()
            cancel.set()
            try:
                with self.assertRaises(S.ClientGone):
                    vision.encode_prompt("<__media__>", [Path(tmp)], 1, cancel)
            finally:
                vision.shutdown()

    def test_common_engine_writes_standard_geni_without_dropping_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            script = path / "engine"
            capture = path / "commands.txt"
            script.write_text("#!" + sys.executable + "\n" +
                              "import pathlib,sys\nprint('READY 256 stop',flush=True)\n" +
                              "for line in sys.stdin:\n" +
                              "    pathlib.Path(" + repr(str(capture)) + ").write_text(line)\n" +
                              "    if line.startswith('QUIT'): break\n" +
                              "    print('T 65',flush=True)\n" +
                              "    print('DONE 1 3 1 1 stop 0 0 0 0 0',flush=True)\n")
            script.chmod(0o700)
            engine = M.M31Engine(str(script), [])
            try:
                ids = M.PreparedIds([41, -1, 42], {"vision_prompt_rows": 3})
                output = str(path / "image.sve")
                self.assertEqual(list(engine.generate(ids, 1, {}, threading.Event(), output)), [65])
                line = capture.read_text()
                self.assertTrue(line.startswith("GENI 1 "), line)
                self.assertTrue(line.endswith(" " + output + " 41,-1,42\n"), line)
                self.assertIn("chat_b64=", line)
            finally:
                engine.close()


class Template:
    def render(self, messages, **kwargs):
        return "prompt:" + "".join(message.get("content", "") for message in messages)


class FakeVision:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.media = MediaInput(directory)
        self.failure = False
        self.manifest, _ = fixture(directory)
        self.media._read = mock.Mock(return_value=b"test")
        self.media._store = mock.Mock(return_value=Path("test.png"))

    def encode_prompt(self, prompt, media, fps, cancel=None):
        if self.failure:
            raise S.VisionUnavailable("vision unavailable: helper stopped")
        self.prompt, self.sources, self.fps = prompt, media, fps
        return self.manifest


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vision = FakeVision(self.temp.name)
        self.engine = S.MockEngine(S.ByteTokenizer(), "ok", max_context=256)
        self.service = M.M31Service(self.engine, S.ByteTokenizer(), Template(), vision=self.vision)
        self.message = {"role": "user", "content": [{"type": "text", "text": "look"},
                                                    {"type": "image", "source": "test.png"}]}
        self.addCleanup(self.service.drop_embeddings)

    def test_prepared_native_metadata_and_common_embedding_lifecycle(self):
        ids, thinking, maximum = self.service.prepare([self.message], [], {}, 20,
                                                      req={"cache_prompt": False})
        self.assertEqual(ids, [41, -1, -1, 42, -1, 43])
        self.assertIn("<__media__>", self.vision.prompt)
        self.assertFalse(ids.cache_prompt)
        self.assertEqual(ids.chat_request["vision_prompt_rows"], 6)
        native_text = Template().render(ids.chat_request["messages"])
        self.assertEqual(ids.chat_request["vision_template_ids"], self.service.tok.encode(native_text, parse_special=True))
        path = self.service.embeddings.path
        self.assertTrue(path.is_file())
        self.assertFalse(self.service.fifo.locked())
        self.service.drop_embeddings()
        self.assertFalse(path.exists())

    def test_forced_prefix_replays_after_mixed_rows(self):
        ids, _, _ = self.service.prepare([self.message], [], {"enable_thinking": False}, 20, force="forced")
        self.assertEqual(ids.chat_request["vision_prompt_rows"], 6)
        self.assertEqual(ids[6:], self.service.tok.encode("forced", parse_special=True))

    def test_failed_helper_releases_fifo_and_next_text_prepare_works(self):
        self.vision.failure = True
        with self.assertRaisesRegex(S.VisionUnavailable, "vision unavailable"):
            self.service.prepare([self.message], [], {}, 20)
        self.assertFalse(self.service.fifo.locked())
        self.assertEqual(list(self.vision.dir.glob("request-*.sve")), [])
        ids, _, _ = self.service.prepare([{"role": "user", "content": "text"}], [], {}, 20)
        self.assertNotIn(-1, ids)
        self.assertNotIn("vision_prompt_rows", ids.chat_request)
        self.assertIsNone(self.service.embeddings.path)

    def test_rejected_context_does_not_leave_embedding_file(self):
        self.engine.max_context = 5
        with self.assertRaises(S.VisionUnavailable):
            self.service.prepare([self.message], [], {}, 20)
        self.assertEqual(list(self.vision.dir.glob("request-*.sve")), [])

    def test_http_helper_failure_returns_503_then_text_still_works(self):
        self.vision.failure = True
        server = S.serve(self.service, port=0)
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        url = "http://127.0.0.1:" + str(server.server_address[1]) + "/v1/chat/completions"
        def post(content):
            request = urllib.request.Request(url, json.dumps({"messages": [{"role": "user", "content": content}],
                                                             "max_tokens": 2}).encode(),
                                             {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=3) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as exc:
                with exc:
                    return exc.code, json.load(exc)
        status, result = post([{"type": "image_url", "image_url": {"url": "test.png"}}])
        self.assertEqual(status, 503, result)
        self.assertIn("vision unavailable", result["error"]["message"])
        status, result = post("still text")
        self.assertEqual(status, 200, result)
        self.assertFalse(self.service.fifo.locked())


if __name__ == "__main__":
    unittest.main()
