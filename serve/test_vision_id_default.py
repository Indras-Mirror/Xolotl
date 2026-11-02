"""Load a real server config with a mock engine, then exercise template defaults and prompt encoding."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from serve.frontend import openai_to_messages
from serve import server


class ReachedServe(Exception):
    pass


class VisionIdDefaults(unittest.TestCase):
    def service(self, cfg):
        seen = []

        def reached(svc, **kwargs):
            seen.append(svc)
            raise ReachedServe

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(cfg))
            with mock.patch.object(sys, 'argv', ['server.py', '--engine', 'mock', '--port', '0', '--config', str(path)]), \
                    mock.patch.object(server, 'serve', reached), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(ReachedServe):
                    server.main()
        return seen[0]

    def request(self, options=None):
        req = {'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'first.png'}},
            {'type': 'text', 'text': ' then '},
            {'type': 'image_url', 'image_url': {'url': 'second.png'}},
        ]}], 'chat_template_kwargs': {'enable_thinking': False, **(options or {})}}
        return openai_to_messages(req)

    def test_absent_false_and_invalid_config_keep_the_old_prompt(self):
        messages, tools, kwargs = self.request()
        plain = self.service({}).render_prompt(messages, tools, kwargs)
        for value in (False, None, 'true', 'false', 0, 1):
            with self.subTest(value=value):
                self.assertEqual(self.service({'add_vision_id': value}).render_prompt(messages, tools, kwargs), plain)

    def test_loaded_config_applies_labels_and_boolean_request_overrides_it(self):
        enabled = self.service({'add_vision_id': True})
        disabled = self.service({'add_vision_id': False})
        messages, tools, kwargs = self.request()
        plain = disabled.render_prompt(messages, tools, kwargs)
        labeled = enabled.render_prompt(messages, tools, kwargs)
        self.assertIn('Picture 1: <|vision_start|>', labeled)
        self.assertIn('Picture 2: <|vision_start|>', labeled)
        self.assertEqual(labeled.replace('Picture 1: ', '').replace('Picture 2: ', ''), plain)
        self.assertEqual(enabled.render_prompt(*self.request({'add_vision_id': False})), plain)
        self.assertEqual(disabled.render_prompt(*self.request({'add_vision_id': True})), labeled)

    def test_labels_change_encoded_prompt_and_explicit_false_restores_it(self):
        svc = self.service({'add_vision_id': True})
        messages, tools, off = self.request({'add_vision_id': False})
        _, _, default = self.request()
        plain = svc.encode_prompt(messages, tools, off)
        labeled = svc.encode_prompt(messages, tools, default)
        self.assertNotEqual(plain, labeled)
        self.assertIn('Picture 1:', svc.tok.decode(labeled))
        self.assertEqual(svc.encode_prompt(messages, tools, off), plain)
        pad = svc.tok.encode('<|image_pad|>', parse_special=True)[0]
        self.assertEqual(plain.count(pad), labeled.count(pad))


if __name__ == '__main__':
    unittest.main()
