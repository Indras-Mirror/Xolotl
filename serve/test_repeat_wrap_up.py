"""CPU-only regression: a one-token run inside the thinking closes the thinking instead of ending the reply
(python -m unittest serve.test_repeat_wrap_up).  GLM-5.3-Flash wrote 'Dot' 256 times at 18,361 thinking tokens
(2026-11-09) and the reply ended as "length" with no answer."""
import threading
import unittest
from serve.server import Service, HIGH_EFFORT
from serve.test_reasoning_loop_recovery import Tokenizer, RecordingEngine


class RepeatWrapUp(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.ids = self.tok.encode('<|im_start|>system\n' + HIGH_EFFORT +
                                   '<|im_end|>\n<|im_start|>user\nSolve it<|im_end|>\n<|im_start|>assistant\n<think>\n',
                                   parse_special=True)
        self.thinking = 'Working through the cases. ' + 'x' * 300

    def run_case(self, wrap_up, script):
        engine = RecordingEngine(self.tok, script)
        service = Service(engine, self.tok, None)
        service.answer_room = False
        service.repeat_wrap_up = wrap_up
        events = list(service.run(self.ids, True, [], 4000, {}, threading.Event()))
        text = ''.join(getattr(ev, 'text', '') or '' for kind, ev in events if kind == 'event' and ev.kind == 'content')
        return engine, text

    def test_a_run_in_the_thinking_closes_it_and_the_model_answers(self):
        engine, text = self.run_case(True, [self.thinking, 'The answer is 42.'])
        self.assertEqual(len(engine.calls), 2)
        second = self.tok.decode(engine.calls[1][0])
        self.assertIn('</think>', second)                         # the thinking was closed
        self.assertNotIn('x' * 8, second)                         # the run is not fed back to the engine
        self.assertIn('Working through the cases.', second)       # the thinking before it is
        self.assertIn('The answer is 42.', text)

    def test_off_ends_the_reply_as_before(self):
        engine, text = self.run_case(False, [self.thinking, 'The answer is 42.'])
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(text, '')

    def test_only_once_a_reply(self):
        engine, _ = self.run_case(True, [self.thinking, 'y' * 300, 'never'])
        self.assertEqual(len(engine.calls), 2)                    # a second run (in the answer) ends the reply

    def test_a_run_in_the_answer_still_ends_it(self):
        engine, _ = self.run_case(True, ['Short thought.\n</think>\nAnswer: ' + 'z' * 300, 'never'])
        self.assertEqual(len(engine.calls), 1)


if __name__ == '__main__':
    unittest.main()
