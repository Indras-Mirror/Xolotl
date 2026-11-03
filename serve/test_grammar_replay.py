"""Grammar replay across the real scheduler's GEN/BGEN/GEN migrations, CPU pipe only."""
import json
import threading
import time
from unittest import mock
from serve import test_parallel as P


class GrammarReplay(P.FakeEngineCase):
    def migration(self, capability):
        source = P.FAKE_BATCH.replace('print("INFO engine=0.1.39"',
            'print("INFO engine=0.1.39 grammar=1' + (' grammar_replay=1' if capability else '') + '"')
        source = source.replace('log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\\n"); log.flush()',
            'log.write(__import__("json").dumps({"line":line}) + "\\n"); log.flush()')
        with mock.patch.object(P, 'FAKE_BATCH', source):
            self.start(2, slot_cache=True)
        initial = list(b'LONGREPLY please')
        base = [900, 901]
        sampling = {'_grammar_path': '/synthetic.gbnf', '_grammar_prefix': base, '_grammar_after': 902}
        outputs, errors = {}, []
        def long():
            try:
                outputs['long'] = [t for t in self.engine.generate(initial, 200, sampling, threading.Event()) if t is not None]
            except BaseException as e:
                errors.append(e)
        worker = threading.Thread(target=long)
        worker.start()
        # Wait for actual first response delivery, then overlap the shorter request.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.log.exists() and self.log.read_text():
                break
            time.sleep(.005)
        time.sleep(.05)
        outputs['short'] = [t for t in self.engine.generate(list(b'short'), 200, sampling, threading.Event()) if t is not None]
        worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(outputs['long'], list(b'ok, ' + b'la ' * 15 + b'done.') + [257])
        self.assertEqual(outputs['short'], list(b'ok, done.') + [257])
        records = [json.loads(line)['line'].split() for line in self.log.read_text().splitlines() if line.startswith('{')]
        long_legs = [f for f in records if [int(v) for v in f[-1].split(',')][:len(initial)] == initial]
        self.assertEqual(long_legs[0][0], 'GEN')
        self.assertIn('BGEN', [f[0] for f in long_legs])
        self.assertEqual(long_legs[-1][0], 'GEN', long_legs)
        for fields in long_legs:
            keys = dict(f.split('=', 1) for f in fields[1:-1] if '=' in f)
            self.assertEqual(keys['grammar_after'], '902')
            suffix = [int(v) for v in fields[-1].split(',')][len(initial):]
            self.assertEqual(suffix, outputs['long'][:len(suffix)])
            if capability:
                self.assertEqual([int(v) for v in keys['grammar_prefix'].split(',')], base + suffix)
            else:
                self.assertNotIn('grammar_prefix', keys)
        self.assertEqual(sampling['_grammar_prefix'], base, 'continuations mutated original segment prefix')
        self.assertFalse(any(self.engine.slot_busy))
        self.assertTrue(self.engine.alive())

    def test_explicit_prefix_survives_solo_batch_solo_without_duplicates(self):
        self.migration(True)

    def test_capability_absent_preserves_original_protocol(self):
        self.migration(False)
