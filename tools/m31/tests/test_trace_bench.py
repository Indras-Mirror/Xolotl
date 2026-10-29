#!/usr/bin/env python3
"""Model-free checks for the optional protocol tee and actual-chunk parser."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
from types import SimpleNamespace
import unittest

STAGED_ROOT = Path(__file__).resolve().parents[3]
SOURCE = STAGED_ROOT


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRACE = load('m31_trace_server_test', STAGED_ROOT / 'tools/m31/bench/trace_server.py')
CHUNKS = load('m31_trace_chunks_test', STAGED_ROOT / 'tools/m31/bench/trace_chunks.py')


class TracePump(unittest.TestCase):
    def test_original_ready_and_pump_preserve_every_line_without_double_tee(self):
        module = TRACE.load_shim(SOURCE)
        output = io.StringIO()
        TRACE.install_trace(module, output)
        original = ["INFO engine=m31\n", "READY 40000 stop\n",
                    "PP 10 20 1 1000\n", "T 5\n", "PP 20 20 2 1000\n", "T 6\n",
                    "DONE 2 20 2 1 length 0 0 0 0 0 0 0 0 20 0\n"]
        pipe = io.StringIO(''.join(original))
        proc = SimpleNamespace(stdout=pipe, pid=12345)
        ready = queue.Queue()
        module.M31Engine._ready_pump(proc, ready)
        self.assertEqual([ready.get_nowait(), ready.get_nowait()], original[:2])
        tee = proc.stdout
        worker = SimpleNamespace(proc=proc, lines=queue.Queue(), slot_q=[], ended=False)
        module.M31Engine._pump(worker)
        observed = []
        while not worker.lines.empty():
            observed.append(worker.lines.get_nowait())
        self.assertEqual(observed, original[2:] + [None])
        self.assertIs(proc.stdout, tee)
        self.assertIs(tee.original, pipe)
        self.assertTrue(worker.ended)
        logged = [json.loads(line)['line'] for line in output.getvalue().splitlines()]
        self.assertEqual(logged, [line.rstrip('\n') for line in original if not line.startswith('T ')])
        self.assertFalse(any(line.startswith('T ') for line in logged))
        self.assertEqual(len(logged), 5)
        proc.stdout.close()
        self.assertTrue(pipe.closed)

    def test_readline_returns_original_text_and_omits_token_records(self):
        output = io.StringIO()
        tee = TRACE.TeeReader(io.StringIO('T 8\nERR fixture refusal\n'), TRACE.ControlTrace(output), 321)
        self.assertEqual(tee.readline(), 'T 8\n')
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(tee.readline(), 'ERR fixture refusal\n')
        self.assertEqual([json.loads(line)['line'] for line in output.getvalue().splitlines()], ['ERR fixture refusal'])
        self.assertEqual(tee.readline(), '')


class ChunkRecipes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.trace, self.requests = self.root / 'trace.jsonl', self.root / 'raw.jsonl'
        self.rows, self.raw = [], []
        for depth, n, generated in [('warmup', 1200, 32), ('4K', 4346, 256),
                                    ('16K', 16631, 256), ('32K', 33019, 256)]:
            pieces = [n] if n < 5000 else [4096] * (n // 4096) + ([n % 4096] if n % 4096 else [])
            position = 0
            for size in pieces:
                position += size
                self.rows.append({'pid': 12345, 'line': f'PP {position} {n} 1 1'})
            self.rows.append({'pid': 12345, 'line': f'DONE {generated} {n} 1 1 length 0 0 0 0 0 0 0 0 {n} 0'})
            self.raw.append({'depth': depth, 'request': {'cache_prompt': False},
                             'response': {'usage': {'prompt_tokens': n}}})
        self.requests.write_text(''.join(json.dumps(row) + '\n' for row in self.raw))
        self.write(self.rows)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows):
        self.trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def parse(self):
        return CHUNKS.parse_groups(self.trace, self.requests)

    def test_complete_actual_chunks_match_done_and_api_totals(self):
        result = self.parse()
        self.assertEqual([row['depth'] for row in result], ['warmup', '4K', '16K', '32K'])
        self.assertEqual(result[-1]['actual_chunks'], [4096] * 8 + [251])
        self.assertEqual(sum(result[-1]['actual_chunks']), 33019)
        self.assertTrue(all(row['read'] == row['prompt_n'] and row['reused'] == 0 for row in result))

    def test_missing_prompt_progress_is_rejected(self):
        self.write(self.rows[1:])
        with self.assertRaisesRegex(ValueError, 'PP chunks'):
            self.parse()

    def test_changed_prompt_progress_total_is_rejected(self):
        bad = copy.deepcopy(self.rows)
        bad[0]['line'] = 'PP 1200 1300 1 1'
        self.write(bad)
        with self.assertRaisesRegex(ValueError, 'continuity/total'):
            self.parse()

    def test_reused_prefix_in_uncached_benchmark_is_rejected(self):
        self.write([{'pid': 12345, 'line': 'RESUME 256'}, *self.rows])
        with self.assertRaisesRegex(ValueError, 'reused or omitted'):
            self.parse()

    def test_hidden_engine_restart_is_rejected(self):
        bad = copy.deepcopy(self.rows)
        bad[1]['pid'] = 54321
        self.write(bad)
        with self.assertRaisesRegex(ValueError, 'restarted'):
            self.parse()

    def test_token_record_in_control_trace_is_rejected(self):
        self.write([{'pid': 12345, 'line': 'T 2'}, *self.rows])
        with self.assertRaisesRegex(ValueError, 'non-control'):
            self.parse()

    def test_engine_error_is_not_reported_as_a_completed_recipe(self):
        self.write([{'pid': 12345, 'line': 'ERR fixture unavailable'}, *self.rows])
        with self.assertRaisesRegex(ValueError, 'engine ERR'):
            self.parse()

    def test_output_budget_differs_from_request_is_rejected(self):
        bad = copy.deepcopy(self.rows)
        bad[1]['line'] = 'DONE 31 1200 1 1 length 0 0 0 0 0 0 0 0 1200 0'
        self.write(bad)
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.parse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=STAGED_ROOT,
                        help='actual checkout for the model-free original pump test')
    args, rest = parser.parse_known_args()
    SOURCE = args.source.resolve(strict=True)
    unittest.main(argv=[__file__, *rest])
