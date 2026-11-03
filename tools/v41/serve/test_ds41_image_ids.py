"""Real DeepSeek template + Service UUID resolution, with a CPU keyed encoder."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_ds41 as D
from serve.test_image_ids import KeyedVision, RefEngine, chat, png


class DsTokenizer(D.S.ByteTokenizer):
    SPECIALS = D.S.ByteTokenizer.SPECIALS + [D.E.eos_token, D.E.IMAGE_PLACEHOLDER]


class DsImageIds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tok = DsTokenizer()
        self.vision = KeyedVision(self.tmp.name)
        self.engine = RefEngine(self.tok, 'ok', max_context=32768)
        self.template = D.Ds41Template()
        self.svc = D.S.Service(self.engine, self.tok, self.template, vision=self.vision)

    def tearDown(self):
        self.svc.drop_embeddings()
        self.tmp.cleanup()

    def prepare(self, *parts):
        messages, tools, kwargs = chat(*parts)
        before = copy.deepcopy(messages)
        ids, _, _ = self.svc.prepare(messages, tools, kwargs, 8)
        self.assertEqual(messages, before)
        return ids

    def test_known_uuid_null_keeps_prompt_and_avoids_load_and_encode(self):
        data = {'type': 'image_url', 'image_url': {'url': png(1)}, 'uuid': 'frame'}
        original_ids = self.prepare(data)
        self.assertEqual(self.vision.encodes, 1)
        self.svc.drop_embeddings()
        with patch.object(D.S.Vision, 'load', side_effect=AssertionError('reference reread image bytes')):
            ids = self.prepare({'type': 'image_url', 'image_url': None, 'uuid': 'frame'})
        self.assertEqual(ids, original_ids)
        self.assertEqual(self.vision.encodes, 1)
        self.assertEqual(ids.count(self.tok.encode(D.E.IMAGE_PLACEHOLDER, True)[0]), 4)

    def test_unknown_uuid_keeps_service_cache_miss_wording(self):
        with self.assertRaisesRegex(D.S.ImageCacheMiss,
                '^Cache miss for image at index 0 but data is not provided\\.$'):
            self.prepare({'type': 'image_url', 'image_url': None, 'uuid': 'unknown'})
        self.assertEqual(self.vision.encodes, 0)
        self.assertNotIn('unknown', self.svc.image_ids)

    def test_mixed_images_preserve_order_and_original_sources(self):
        a = {'type': 'image_url', 'image_url': {'url': png(1)}, 'uuid': 'a'}
        b = {'type': 'image_url', 'image_url': {'url': png(2)}, 'uuid': 'b'}
        ids = self.prepare(a, b)
        self.svc.drop_embeddings()
        replay = self.prepare({'type': 'image_url', 'image_url': None, 'uuid': 'a'}, b)
        self.assertEqual(replay, ids)
        self.assertEqual(self.vision.encodes, 2)

    def test_template_reference_is_stable_and_cannot_bypass_missing_source_validation(self):
        messages, _, _ = chat({'type': 'image_url', 'image_url': None, 'uuid': 'unknown'})
        original = copy.deepcopy(messages)
        first = self.template.render(messages)
        self.assertEqual(first, self.template.render(messages))
        self.assertEqual(first.count(D.E.IMAGE_PLACEHOLDER), 1)
        self.assertEqual(messages, original)
        with self.assertRaisesRegex(ValueError, 'Image block does not contain a valid source'):
            self.template.render([{'role': 'user', 'content': [{'type': 'image', 'source': ''}]}])
        with self.assertRaisesRegex(ValueError, 'image placeholder'):
            self.template.render([{'role': 'user', 'content': [{'type': 'text', 'text': D.E.IMAGE_PLACEHOLDER},
                {'type': 'image', 'source': 'real-source'}]}])


if __name__ == '__main__':
    unittest.main()
