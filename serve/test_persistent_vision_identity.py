"""CPU identity and HAVE ordering regressions; no native/GPU inference is claimed."""
import base64
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from serve.server import (ROOT, ByteTokenizer, Service, Vision, bind_persistent_vision, child_env)
from serve.frontend import ChatTemplate
from serve.test_image_ids import KeyedVision, RefEngine, png


class BoundVision(KeyedVision):
    persistence_namespace = Vision.persistence_namespace

    def __init__(self, directory, config):
        super().__init__(directory)
        self.spawn = (Vision.reader_args(config), None, {})
        self.proc = SimpleNamespace(pid=os.getpid())
        self.persistent_images = self.persistence_requested = True
        self.on_start = None

    def alive(self):
        return True


class PersistentVisionIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.files = {}
        for name in ('reader', 'cpu-reader', 'projector', 'model'):
            file = self.root / name
            file.write_bytes(b'immutable fixture')
            file.chmod(0o700)
            self.files[name] = str(file)
        self.config = dict(exe=self.files['reader'], mmproj=self.files['projector'],
                           model=self.files['model'], gpu=True, min_tokens=20, max_tokens=1024)
        self.raw = base64.b64decode(png(1).split(',', 1)[1])

    def tearDown(self):
        self.tmp.cleanup()

    def service(self, config):
        vision = BoundVision(self.root, config)
        tok = ByteTokenizer()
        engine = RefEngine(tok, 'ok', max_context=32768)
        service = Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'), vision=vision)
        return service, vision, engine

    def test_same_actual_pipeline_reuses_native_picture_after_frontend_restart(self):
        for uid in ('same-frame', None):
            with self.subTest(uuid=uid):
                first, _, engine = self.service(self.config)
                first._pictures([self.raw], [uid])
                key = engine.asked[-1][0]
                second, vision, restarted_engine = self.service(self.config)
                restarted_engine.held.add(key)
                _, refs = second._pictures([self.raw], [uid])
                self.assertEqual(restarted_engine.asked[-1], [key])
                self.assertIsNotNone(refs[0])
                self.assertEqual(vision.encodes, 0)

    def test_changed_pipeline_misses_before_have_for_uuid_and_bytes(self):
        variants = [dict(min_tokens=64), dict(max_tokens=512), dict(gpu=False),
                    dict(exe=self.files['cpu-reader'])]
        for uid in ('same-frame', None):
            for change in variants:
                with self.subTest(uuid=uid, change=change):
                    old, _, engine = self.service(self.config)
                    old._pictures([self.raw], [uid])
                    old_key = engine.asked[-1][0]
                    new, vision, new_engine = self.service(dict(self.config, **change))
                    new_engine.held.add(old_key)
                    _, refs = new._pictures([self.raw], [uid])
                    self.assertNotEqual(new_engine.asked[-1], [old_key])
                    self.assertIsNone(refs[0])
                    self.assertEqual(vision.encodes, 1)

    def test_projector_and_helper_same_size_replacement_invalidates_identity(self):
        for name in ('projector', 'reader'):
            with self.subTest(input=name):
                old, _, engine = self.service(self.config)
                old._pictures([self.raw], ['frame'])
                key = engine.asked[-1][0]
                path = Path(self.files[name]); old_time = path.stat().st_mtime_ns
                path.write_bytes(b'changed!! fixture')
                os.utime(path, ns=(old_time, old_time))
                new, vision, new_engine = self.service(self.config)
                new_engine.held.add(key)
                new._pictures([self.raw], ['frame'])
                self.assertNotEqual(new_engine.asked[-1], [key])
                self.assertEqual(vision.encodes, 1)

    def test_actual_selected_fallback_not_configured_primary_controls_identity(self):
        primary = BoundVision(self.root, self.config)
        fallback = BoundVision(self.root, dict(self.config, exe=self.files['cpu-reader'], gpu=False))
        self.assertNotEqual(primary.persistence_namespace(), fallback.persistence_namespace())
        self.assertEqual(primary.persistence_namespace(), BoundVision(self.root, self.config).persistence_namespace())
        selected = BoundVision(self.root, self.config)
        cpu_config = dict(self.config, exe=self.files['cpu-reader'], gpu=False)
        selected.fallback = (Vision.reader_args(cpu_config), None, {})
        attempts = []
        def start_reader():
            attempts.append(selected.spawn[0][0])
            if selected.spawn[0][0] == self.files['reader']:
                raise RuntimeError('primary reader unavailable')
            selected.proc = SimpleNamespace(pid=os.getpid())
        selected._start_reader = start_reader
        Vision._start(selected)
        self.assertEqual(attempts, [self.files['reader'], self.files['cpu-reader']])
        self.assertEqual(selected.persistence_namespace(), fallback.persistence_namespace())

    def test_split_reader_model_inventory_includes_other_shards(self):
        first = self.root / 'vision-00001-of-00002.gguf'
        second = self.root / 'vision-00002-of-00002.gguf'
        first.write_bytes(b'first'); second.write_bytes(b'second')
        config = dict(self.config, model=str(first))
        before = BoundVision(self.root, config).persistence_namespace()
        second.write_bytes(b'edited')
        self.assertNotEqual(before, BoundVision(self.root, config).persistence_namespace())
        second.unlink()
        with self.assertRaises(FileNotFoundError):
            BoundVision(self.root, config).persistence_namespace()

    def test_split_snapshot_symlink_inventory_uses_configured_shard_names(self):
        snapshot = self.root / 'snapshot'; snapshot.mkdir()
        blobs = self.root / 'blobs'; blobs.mkdir()
        one = blobs / 'opaque-a'; two = blobs / 'opaque-b'
        one.write_bytes(b'first'); two.write_bytes(b'second')
        first = snapshot / 'vision-00001-of-00002.gguf'
        first.symlink_to(one)
        (snapshot / 'vision-00002-of-00002.gguf').symlink_to(two)
        config = dict(self.config, model=str(first))
        before = BoundVision(self.root, config).persistence_namespace()
        two.write_bytes(b'edited')
        self.assertNotEqual(before, BoundVision(self.root, config).persistence_namespace())
        two.unlink()
        with self.assertRaises(FileNotFoundError):
            BoundVision(self.root, config).persistence_namespace()

    def test_cache_log_paths_and_unselected_config_do_not_change_actual_pipeline(self):
        one = BoundVision(self.root, self.config)
        two = BoundVision(self.root, self.config)
        one.spawn = (one.spawn[0], None, {'STRATA_PREFIX_CACHE_DIR': '/cache-a', 'STRATA_TEST_LOG': '/log-a'})
        two.spawn = (two.spawn[0], None, {'STRATA_PREFIX_CACHE_DIR': '/cache-b', 'STRATA_TEST_LOG': '/log-b'})
        self.assertEqual(one.persistence_namespace(), two.persistence_namespace())

    def test_cuda_anonymous_deleted_scratch_is_not_a_deleted_library(self):
        vision = BoundVision(self.root, self.config)
        expected = vision.persistence_namespace()
        maps_path = Path('/proc') / str(os.getpid()) / 'maps'
        actual_maps = maps_path.read_text()
        read_text = Path.read_text
        def with_mapping(mapping):
            def read(path, *args, **kwargs):
                if path == maps_path:
                    return actual_maps + '\n1000-2000 rw-s 00000000 00:01 1 ' + mapping + '\n'
                return read_text(path, *args, **kwargs)
            return read
        with patch.object(Path, 'read_text', with_mapping('/dev/zero (deleted)')):
            self.assertEqual(expected, vision.persistence_namespace())
        with patch.object(Path, 'read_text', with_mapping('/usr/lib/libreader.so (deleted)')):
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                vision.persistence_namespace()

    def test_unstarted_lazy_reader_is_not_started_or_bound(self):
        lazy = SimpleNamespace(alive=lambda: False, persistent_images=False,
                               persistence_namespace=lambda: self.fail('lazy reader must stay unresolved'))
        env = {'STRATA_PREFIX_CACHE_DIR': '/cache', 'STRATA_PREFIX_CACHE_VISION_ID': 'stale'}
        bind_persistent_vision(env, lazy)
        self.assertNotIn('STRATA_PREFIX_CACHE_VISION_ID', env)
        self.assertFalse(lazy.persistent_images)
        eager = BoundVision(self.root, self.config)
        bind_persistent_vision(env, eager)
        self.assertEqual(env['STRATA_PREFIX_CACHE_VISION_ID'],
                         'frontend-vision-pipeline-v1:' + eager.persistence_namespace())

    def test_unavailable_startup_inventory_leaves_image_state_unbound(self):
        def unavailable():
            raise OSError('reader maps unavailable')
        vision = SimpleNamespace(alive=lambda: True, persistence_namespace=unavailable,
                                 persistent_images=True)
        env = {'STRATA_PREFIX_CACHE_DIR': '/cache', 'STRATA_PREFIX_CACHE_VISION_ID': 'stale'}
        bind_persistent_vision(env, vision)
        self.assertNotIn('STRATA_PREFIX_CACHE_VISION_ID', env)
        self.assertFalse(vision.persistent_images)

    def test_default_off_keeps_existing_picture_keys_and_does_not_probe_reader(self):
        service, vision, engine = self.service(self.config)
        vision.persistent_images = False
        vision.persistence_namespace = lambda: self.fail('default-off must not fingerprint/start reader')
        service._pictures([self.raw], [None])
        self.assertEqual(engine.asked[-1], [Vision.key_of(Vision.normalize(self.raw))[:16]])
        env = {}
        bind_persistent_vision(env, vision)
        self.assertEqual(env, {})
        with patch.dict(os.environ, {'STRATA_PREFIX_CACHE_VISION_ID': 'stale'}, clear=False):
            self.assertNotIn('STRATA_PREFIX_CACHE_VISION_ID', child_env({}))


if __name__ == '__main__':
    unittest.main()
