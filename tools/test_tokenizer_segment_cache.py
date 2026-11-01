"""Exact memoization boundaries, bounded retention and concurrent caller isolation."""
import concurrent.futures
import contextlib
import io
import json
import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST


def tokenizer(budget=1 << 20, pre="qwen35", merges=True, cls=ST.Tokenizer):
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    types = [1] * 256
    rules = ["a b", "ab c", "t h", "th e"] if merges else []
    for pair in rules:
        a, b = pair.split(" ")
        tokens.append(a + b)
        types.append(1)
    tokens += ["<think>", "</think>", "<|im_start|>", "<|im_end|>"]
    types += [4, 4, 3, 3]
    return cls(tokens, rules, types, pre=pre, segment_cache_bytes=budget)


class LegacyMarkedTokenizer(ST.Tokenizer):
    def _encode_plain_marked(self, text, marks, base, out_marks):
        out, a = [], 0
        for piece in self._pre_split(text):
            b = a + len(piece)
            for k, c in marks:
                if base + a <= c < base + b and k not in out_marks:
                    out_marks[k] = (len(out), base + a)
            out.extend(self._encode_plain(piece))
            a = b
        return out


class SegmentCache(unittest.TestCase):
    def test_hit_skips_pre_split_and_returns_fresh_list(self):
        tk = tokenizer()
        text = "abc the café🙂 " * 40
        first = tk.encode(text)
        with patch.object(tk, "_pre_split", side_effect=AssertionError("must hit")):
            second = tk.encode(text)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        second[:] = [-1]
        self.assertEqual(first, tk.encode(text))
        self.assertEqual(tk._segment_ids.stats()["hits"], 2)

    def test_changed_tail_and_appended_merge_boundaries_are_not_prefix_hits(self):
        off, on = tokenizer(0), tokenizer()
        for suffix in ("a", "ab", "abc", "abc ", "abc\n", "abc\n\n", "abc🙂", "abc\u0301"):
            text = "stable text " * 40 + suffix
            self.assertEqual(off.encode(text), on.encode(text))
        self.assertEqual(on._segment_ids.stats()["hits"], 0)
        self.assertEqual(on._segment_ids.stats()["misses"], 8)

    def test_special_and_plain_modes_cannot_poison_each_other(self):
        off, on = tokenizer(0), tokenizer()
        text = "abc " * 90 + "<|im_start|>" + "the " * 90 + "<think>abc</think><|im_end|>"
        for _ in range(3):
            for special in (False, True):
                for spans in ((), [(0, len(text))], [(text.index("<think>"), len(text))]):
                    self.assertEqual(off.encode(text, special, plain=spans),
                                     on.encode(text, special, plain=spans))

    def test_marked_frontiers_and_cached_unmarked_segments_match(self):
        off, on = tokenizer(0), tokenizer()
        text = "abc " * 100 + "<|im_start|>" + "the\n🙂 " * 100
        for marks in ([], [0], [1, 255, 400, 405, len(text) - 1], [len(text)], [21, 21, 0]):
            for special in (False, True):
                self.assertEqual(off.encode_at_offsets(text, marks, special),
                                 on.encode_at_offsets(text, marks, special))
                self.assertEqual(off.encode_at_offsets(text, marks, special, plain=[(0, len(text))]),
                                 on.encode_at_offsets(text, marks, special, plain=[(0, len(text))]))

    def test_marked_hits_rebind_base_ids_and_unsorted_duplicate_offsets(self):
        off, on = tokenizer(0, cls=LegacyMarkedTokenizer), tokenizer()
        text = "abc the café e\u0301🙂\n" * 40
        for base, marks in [(100, [(8, 520), (2, 101), (8, 102), (9, 520)]),
                            (700, [(18, 1120), (22, 701), (18, 702), (19, 1120)]),
                            (700, [(18, 1120), (22, 701), (18, 702), (19, 1120)])]:
            expected, actual = {22: (99, 99)}, {22: (99, 99)}
            a = off._encode_plain_marked(text, marks, base, expected)
            b = on._encode_plain_marked(text, marks, base, actual)
            self.assertEqual(a, b)
            self.assertEqual(expected, actual)
        self.assertGreater(on._segment_ids.stats()["hits"], 0)

    def test_random_marked_hit_frontiers_match_unmodified_algorithm(self):
        rng = random.Random(1031)
        off, on = tokenizer(0, cls=LegacyMarkedTokenizer), tokenizer()
        text = "abc the e\u0301🙂 漢字\n" * 40
        for _ in range(60):
            base = rng.randrange(2000)
            marks = [(rng.randrange(4), base + rng.randrange(-5, len(text) + 5)) for _ in range(6)]
            for repeat in range(2):
                want, got = {9: (2, 3)}, {9: (2, 3)}
                self.assertEqual(off._encode_plain_marked(text, marks, base, want),
                                 on._encode_plain_marked(text, marks, base, got))
                self.assertEqual(want, got)

    def test_plain_and_marked_entries_share_one_budget(self):
        tk = tokenizer(22_000)
        text = "abc the " * 65
        tk.encode(text)
        tk.encode_at_offsets(text, [1, 25, 100])
        self.assertGreater(tk._segment_ids.stats()["entries"], 0)
        self.assertLessEqual(tk._segment_ids.stats()["charged_bytes"], 22_000)
        self.assertEqual(tk._segment_ids.bytes, sum(v[1] for v in tk._segment_ids.entries.values()))

    def test_ineligible_marked_segments_use_legacy_path_without_cache_calls(self):
        for budget, text, count in [(0, "abc " * 100, 4), (1 << 20, "abc", 2),
                                    (1 << 20, "abc " * 100, 17)]:
            tk = tokenizer(budget)
            marks = [(i, i) for i in range(count)]
            expected = {}
            ids = tokenizer(0, cls=LegacyMarkedTokenizer)._encode_plain_marked(text, marks, 0, expected)
            got = {}
            with patch.object(tk._segment_ids, "get", side_effect=AssertionError("must bypass")), \
                 patch.object(tk._segment_ids, "put", side_effect=AssertionError("must bypass")):
                self.assertEqual(ids, tk._encode_plain_marked(text, marks, 0, got))
            self.assertEqual(expected, got)

    def test_both_pre_tokenizers_unicode_and_random_spans(self):
        rng = random.Random(1031)
        pieces = ["abc", "the", " café🙂 ", "漢字", "e\u0301", "12345", "\n\r", "<think>", "<|im_end|>"]
        for pre in ("qwen35", "joyai-llm"):
            off, on = tokenizer(0, pre), tokenizer(1 << 20, pre)
            for _ in range(60):
                text = "".join(rng.choice(pieces) for _ in range(150))
                spans = [(rng.randrange(len(text)), rng.randrange(len(text))) for _ in range(3)]
                for special in (False, True):
                    expected = off.encode(text, special, plain=spans)
                    self.assertEqual(expected, on.encode(text, special, plain=spans))
                    self.assertEqual(expected, on.encode(text, special, plain=spans))

    def test_lru_and_byte_budget(self):
        cache = ST._PlainSegmentCache(12_000)
        texts = [(str(i) + "x") * 150 for i in range(4)]
        for t in texts[:2]:
            cache.put(t, list(range(100)))
        self.assertIsNotNone(cache.get(texts[0]))
        cache.put(texts[2], list(range(100)))
        self.assertIsNone(cache.get(texts[1]))
        self.assertIsNotNone(cache.get(texts[0]))
        self.assertIsNotNone(cache.get(texts[2]))
        for t in texts:
            cache.put(t, list(range(150)))
            self.assertLessEqual(cache.stats()["charged_bytes"], cache.budget)
            self.assertEqual(cache.bytes, sum(v[1] for v in cache.entries.values()))

    def test_oversize_never_evicts_useful_entry(self):
        cache = ST._PlainSegmentCache(6000)
        small, big = "s" * 300, "b" * 3000
        cache.put(small, [1, 2, 3])
        cache.put(big, list(range(500)))
        self.assertIsNotNone(cache.get(small))
        self.assertIsNone(cache.get(big))
        self.assertEqual(cache.stats()["entries"], 1)

    def test_entry_count_bounded_and_short_strings_not_retained(self):
        cache = ST._PlainSegmentCache(1 << 20)
        for i in range(100):
            cache.put(str(i) + "x" * 300, [i])
        cache.put("short", [0])
        self.assertEqual(cache.stats()["entries"], cache.MAX_ENTRIES)
        self.assertIsNone(cache.get("short"))
        self.assertIsNone(cache.get("0" + "x" * 300))

    def test_replacement_accounting(self):
        cache = ST._PlainSegmentCache(6000)
        text = "x" * 300
        cache.put(text, [1, 2, 3])
        old = cache.bytes
        cache.put(text, [1, 2, 3])
        self.assertEqual(old, cache.bytes)
        self.assertEqual(cache.stats()["entries"], 1)

    def test_tokenizer_config_and_instance_isolation(self):
        merged, raw = tokenizer(), tokenizer(merges=False)
        text = "abc the " * 70
        a, b = merged.encode(text), raw.encode(text)
        self.assertNotEqual(a, b)
        self.assertEqual(a, merged.encode(text))
        self.assertEqual(b, raw.encode(text))
        self.assertIsNot(merged._segment_ids, raw._segment_ids)

    def test_concurrent_mutable_outputs_eviction_and_repeat_requests(self):
        off, on = tokenizer(0), tokenizer(20_000)
        texts = ["<|im_start|>" + "abc the café🙂 " * 30 + str(i) + "<|im_end|>" for i in range(9)]
        expected = [off.encode(t, True) for t in texts]
        def run(i):
            k = i % len(texts)
            got = on.encode(texts[k], True)
            self.assertEqual(got, expected[k])
            got[:] = [-1]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run, range(180)))
        self.assertLessEqual(on._segment_ids.stats()["charged_bytes"], on._segment_ids.budget)

    def test_environment_default_optin_and_invalid_value(self):
        tk = tokenizer()
        args = (tk.tokens, list(" ".join(p) for p in tk.ranks), tk.token_types)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ST.Tokenizer(*args)._segment_ids.budget, 0)
        with patch.dict(os.environ, {"STRATA_TOKENIZER_SEGMENT_CACHE_MIB": "2"}):
            self.assertEqual(ST.Tokenizer(*args)._segment_ids.budget, 2 << 20)
        for bad in ("-1", "nan", "1.5"):
            with patch.dict(os.environ, {"STRATA_TOKENIZER_SEGMENT_CACHE_MIB": bad}):
                with self.assertRaises(ValueError):
                    ST.Tokenizer(*args)

    def test_prompt_encoder_actual_current_api_and_interleaved_histories(self):
        rng = random.Random(1101)
        off, on = tokenizer(0), tokenizer()
        enc = ST.PromptEncoder(on, keep=4)
        histories = ["abc the café🙂 " * 45 + str(i) for i in range(7)]
        for turn in range(100):
            caller = rng.randrange(len(histories))
            # Early edits defeat PromptEncoder's prefix reuse; unchanged later
            # stretches still have exactly the same complete BPE input.
            text = ("<|im_start|>system\n" + str(turn % 3) + "<|im_end|>" +
                    "<|im_start|>user\n" + histories[caller] + "<|im_end|>" +
                    "<|im_start|>assistant\n" + "the " * (turn % 9))
            spans = [(text.index("café"), text.index("café") + 4)]
            spans_before = list(spans)
            expected = off.encode(text, True, plain=spans)
            # Current encode_marked produces resume boundaries, whereas the
            # fork's former API is now encode_at_offsets.
            self.assertEqual(off.encode_marked(text, True, spans), on.encode_marked(text, True, spans))
            for store in (False, True):
                got = enc.encode(text, plain=spans, store=store)
                self.assertEqual(got, expected)
                got[:] = [-1]
            self.assertEqual(spans, spans_before)
            if turn % 11 == 0:
                enc.forget(expected)
            histories[caller] += rng.choice(("abc", "\n", "🙂", "<think>"))
        self.assertGreater(on._segment_ids.stats()["hits"], 0)

    def test_prompt_encoder_concurrent_calls_and_caller_mutation(self):
        off, on = tokenizer(0), tokenizer(50_000)
        enc = ST.PromptEncoder(on, keep=3)
        texts = ["<|im_start|>user\n" + "abc the 🙂 " * 50 + str(i) + "<|im_end|>" for i in range(12)]
        expected = [off.encode(t, True) for t in texts]
        def run(i):
            k = i % len(texts)
            got = enc.encode(texts[k], store=bool(i % 3))
            self.assertEqual(got, expected[k])
            got[:] = [-1]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run, range(160)))
        self.assertLessEqual(on._segment_ids.stats()["charged_bytes"], 50_000)

    def test_errors_do_not_publish_partial_segments(self):
        off, on = tokenizer(0), tokenizer()
        text = "abc " * 80 + "\ud800"
        for tk in (off, on):
            for _ in range(2):
                with self.assertRaises(UnicodeEncodeError):
                    tk.encode(text)
        self.assertEqual(on._segment_ids.stats()["entries"], 0)
        self.assertEqual(on.encode("abc " * 80), off.encode("abc " * 80))

    def test_trace_reports_only_numeric_cumulative_metadata(self):
        with patch.dict(os.environ, {"STRATA_TOKENIZER_SEGMENT_CACHE_TRACE": "1"}):
            tk = tokenizer()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            tk.encode("abc " * 100)
            tk.encode("abc " * 100)
        rows = [json.loads(line.split(" ", 1)[1]) for line in buf.getvalue().splitlines()]
        self.assertEqual(rows[-1]["hits"], 1)
        self.assertEqual(set(rows[-1]), {"budget_bytes", "charged_bytes", "entries", "hits", "misses"})
        self.assertTrue(all(isinstance(v, int) for row in rows for v in row.values()))


if __name__ == "__main__":
    unittest.main()
