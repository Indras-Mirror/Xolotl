"""Request-local grammar terminal markers over a real process pipe, without a GPU."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from serve.server import EngineGrammarStarved, StrataEngine

FAKE = r'''
import sys
mode = sys.argv[sys.argv.index('--mode') + 1]
batch = mode != 'solo'
print('INFO grammar=1 grammar_replay=1' + (' batch_slots=2 batch_pinned=1' if batch else ''), flush=True)
print('READY 4096 stop', flush=True)
first = True
for line in sys.stdin:
    if line.startswith('QUIT'): break
    if not line.startswith(('GEN ', 'BGEN ')): continue
    f = line.split()
    slot = int(f[1]) if f[0] == 'BGEN' else None
    if not first:
        print('T 90\nDONE 1 1 0 1 stop', flush=True)
        if slot is not None: print(f'BADM {slot} 0', flush=True)
        continue
    first = False
    print('T 65', flush=True)
    if mode == 'slot':
        print(f'DONE 1 1 0 1 length\nBADM {slot} 1\nBT {slot} 66\nBDONE {slot} 2 grammar 1', flush=True)
    else:
        print('DONE 1 1 0 1 grammar', flush=True)
        if slot is not None: print(f'BADM {slot} 0', flush=True)
'''


class GrammarStarvation(unittest.TestCase):
    def engine(self, mode):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        script = root / 'engine.py'
        if os.name == 'nt':
            script.write_text(FAKE)
            command = root / 'engine.cmd'
            command.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
        else:
            script.write_text('#!' + sys.executable + '\n' + FAKE)
            script.chmod(0o700)
            command = script
        args = ['--mode', mode] + (['--batch', '2'] if mode != 'solo' else [])
        engine = StrataEngine(str(command), args)
        self.addCleanup(engine.close)
        return engine

    def assert_recovery(self, mode):
        engine = self.engine(mode)
        process = engine.proc
        output = []
        with self.assertRaisesRegex(EngineGrammarStarved, 'grammar permits no next token'):
            for token in engine.generate([1], 8, {}, threading.Event()):
                if token is not None:
                    output.append(token)
        self.assertEqual(output, [65, 66] if mode == 'slot' else [65])
        self.assertTrue(engine.alive())
        self.assertIs(engine.proc, process)
        self.assertFalse(any(engine.slot_busy))
        self.assertTrue(all(not held for held in engine.slot_held))
        self.assertEqual(list(engine.generate([2], 8, {}, threading.Event())), [90])
        self.assertIs(engine.proc, process, 'a grammar error must not restart the engine')

    def test_solo_terminal_preserves_next_request(self):
        self.assert_recovery('solo')

    def test_admission_terminal_is_consumed_through_badm(self):
        self.assert_recovery('admit')

    def test_slot_terminal_releases_only_its_request(self):
        self.assert_recovery('slot')

    def test_replay_field_requires_capability(self):
        sampling = {'_grammar_path': '/temporary.gbnf', '_grammar_prefix': [65, 66]}
        self.assertNotIn('grammar_prefix=', StrataEngine.sampling_keys(sampling))
        self.assertIn(' grammar_prefix=65,66', StrataEngine.sampling_keys(sampling, grammar_replay=True))
        with self.assertRaises(ValueError):
            StrataEngine.sampling_keys({**sampling, '_grammar_prefix': [True]}, grammar_replay=True)


if __name__ == '__main__':
    unittest.main()
