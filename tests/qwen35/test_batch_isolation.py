"""Offline parser and orchestration tests: BytesIO/mocked requests, no network/model."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / 'tools' / 'qwen35' / 'check_batch_isolation.py'
spec = importlib.util.spec_from_file_location('isolation_under_test', PATH)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def event(value):
    return ('data: ' + json.dumps(value) + '\n\n').encode()


def valid(text='A public answer.', count=4):
    return event({'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': 'stop'}]}) + event({'choices': [], 'usage': {'prompt_tokens': 20, 'completion_tokens': count, 'total_tokens': 20 + count}}) + b'data: [DONE]\n\n'


def parse(data, budget=64):
    row = {'done': False, 'text': '', 'finish_reason': None, 'usage': None, 'events': []}
    h.parse_sse(io.BytesIO(data), row, budget)
    return row


class Tests(unittest.TestCase):
    def test_valid_public_stream(self):
        self.assertEqual(parse(valid())['text'], 'A public answer.')

    def test_missing_done_rejected(self):
        with self.assertRaises(ValueError): parse(valid().replace(b'data: [DONE]\n\n', b''))

    def test_error_object_rejected(self):
        with self.assertRaises(ValueError): parse(event({'error': {'message': 'injected'}}) + valid())

    def test_reasoning_rejected(self):
        with self.assertRaises(ValueError): parse(event({'choices': [{'delta': {'reasoning_content': 'hidden'}}]}) + valid())

    def test_bad_usage_rejected(self):
        with self.assertRaises(ValueError): parse(valid(count=65))
        with self.assertRaises(ValueError): parse(valid().replace(b'"total_tokens": 24', b'"total_tokens": 99'))

    def test_short_length_rejected(self):
        with self.assertRaises(ValueError): parse(valid().replace(b'"stop"', b'"length"'))

    def test_duplicate_finish_rejected(self):
        with self.assertRaises(ValueError): parse(valid().replace(b'data: [DONE]', event({'choices': [{'delta': {}, 'finish_reason': 'stop'}]}) + b'data: [DONE]'))

    def test_malformed_json_and_choice_rejected(self):
        with self.assertRaises(ValueError): parse(b'data: {bad}\n\n' + valid())
        with self.assertRaises(ValueError): parse(valid().replace(b'"index": 0', b'"index": 1'))

    def test_cases_unconstrained_exact_profiles(self):
        for profile in h.PROFILES:
            cases = h.cases('public-test', profile, 128)
            self.assertEqual(len({case['payload']['messages'][1]['content'] for case in cases}), 4)
            for case in cases:
                self.assertNotIn('grammar', case['payload'])
                self.assertFalse(case['payload']['chat_template_kwargs']['enable_thinking'])
                self.assertEqual(case['encoded'], json.dumps(case['payload'], separators=(',', ':'), ensure_ascii=False).encode())
                self.assertEqual(case['payload']['temperature'], 0 if profile == 'greedy' else .9)

    def test_comparison_marks_divergence_and_ignores_cache_only(self):
        a = {'case': 0, 'request_id': 'a', 'done': True, 'errors': [], 'text': 'abc', 'finish_reason': 'stop', 'usage': {'prompt_tokens': 20, 'completion_tokens': 3, 'total_tokens': 23, 'prompt_tokens_details': {'cached_tokens': 0}}}
        b = {**a, 'request_id': 'b', 'usage': {**a['usage'], 'prompt_tokens_details': {'cached_tokens': 19}}}
        self.assertTrue(h.compare(a, b)['observable_exact_agreement'])
        b['text'] = 'abx'
        self.assertFalse(h.compare(a, b)['observable_exact_agreement'])
        self.assertEqual(h.compare(a, b)['first_difference']['character_offset'], 2)

    def test_real_suite_order_and_body_reuse_without_network(self):
        seen = []
        def fake(url, case, timeout, deadline, request_id, barrier=None):
            if barrier: barrier.wait(timeout=2)
            seen.append((case['case'], id(case['encoded']), case['body_sha256'], request_id))
            return {'case': case['case'], 'request_id': request_id, 'done': True, 'errors': [], 'text': str(case['case']), 'finish_reason': 'stop', 'usage': {'prompt_tokens': 20, 'completion_tokens': 1, 'total_tokens': 21}}
        with tempfile.TemporaryDirectory() as directory, patch.object(h, 'request', fake):
            report = h.run_suite('http://127.0.0.1:1', 'http://127.0.0.1:1', 'public-test', Path(directory) / 'out', 4, (64,), ('sampled',), 1, 10)
        self.assertTrue(report['passed'])
        self.assertEqual(len(seen), 16)
        self.assertTrue(all('-seq-' in row[3] for row in seen[:8]))
        for case in range(4):
            self.assertEqual(len({row[1] for row in seen if row[0] == case}), 1)
            self.assertEqual(len({row[2] for row in seen if row[0] == case}), 1)
        self.assertEqual(report['summary']['seed_repeat_agreements'], 4)
        self.assertEqual(report['summary']['concurrent_agreements'], 8)


if __name__ == '__main__': unittest.main(verbosity=2)
