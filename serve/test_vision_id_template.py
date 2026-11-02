"""Opt-in Qwen picture ordinals must reach the template through the chat API adapter."""
from pathlib import Path
import unittest

from serve.frontend import ChatTemplate, openai_to_messages


class VisionIds(unittest.TestCase):
    def render(self, options):
        req = {'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'first.png'}},
            {'type': 'text', 'text': 'then'},
            {'type': 'image_url', 'image_url': {'url': 'second.png'}},
        ]}], 'chat_template_kwargs': {'enable_thinking': False, **options}}
        messages, tools, kwargs = openai_to_messages(req)
        template = ChatTemplate(Path(__file__).with_name('chat_template.jinja'))
        return template.render(messages, tools, **kwargs)

    def test_opt_in_picture_labels_keep_the_image_order(self):
        plain = self.render({})
        labeled = self.render({'add_vision_id': True})
        marker = '<|vision_start|><|image_pad|><|vision_end|>'
        self.assertIn('Picture 1: ' + marker + 'thenPicture 2: ' + marker, labeled)
        self.assertEqual(labeled.replace('Picture 1: ', '').replace('Picture 2: ', ''), plain)

    def test_default_and_non_booleans_keep_existing_prompt(self):
        plain = self.render({})
        for value in (False, None, 'true', 'false', 1, 0):
            with self.subTest(value=value):
                self.assertEqual(self.render({'add_vision_id': value}), plain)


if __name__ == '__main__':
    unittest.main()
