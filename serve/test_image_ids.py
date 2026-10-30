"""Pictures by id (vLLM's "uuid" on a content part) and pictures the engine already holds (img_refs engines: HAVE and
SVR1 records): a picture is encoded once, a known id needs no bytes, an unknown one is a 400 asking for them."""
import base64
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from serve.frontend import ChatTemplate, openai_to_messages
from serve.server import ROOT, SVR1, ByteTokenizer, MockEngine, Service, Vision

CTX = 32768


def png(seed: int) -> str:
    """A tiny valid PNG (1x1, a colour from `seed`) as a data: URL."""
    raw = b"\x00" + bytes([seed % 256, (seed * 7) % 256, (seed * 13) % 256])
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    body = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(body).decode()


class KeyedVision:
    """An encoder stand-in with keyed lookups: each picture 4 rows (2 x 2) of width 8; counts its encodes."""
    WIDTH = 8

    def __init__(self, d):
        self.dir, self.cache, self.encodes = Path(d), {}, 0

    def lookup(self, key):
        return self.cache.get(key)

    def encode_keyed(self, key, data, keep=()):
        if key in self.cache:
            return self.cache[key]
        self.encodes += 1
        out = self.dir / f"{key}.sve"
        out.write_bytes(struct.pack("<5i", 0x31455653, 4, 2, 2, self.WIDTH) + b"\0" * (4 * 4 * self.WIDTH))
        self.cache[key] = (out, 4)
        return self.cache[key]


class RefEngine(MockEngine):
    """A mock engine that says it takes pictures by reference and holds the ones it was sent."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.info = {"img_refs": 1, "n_embd": KeyedVision.WIDTH}
        self.held: set[str] = set()
        self.asked = []
        self.refill = None

    def alive(self):
        return True

    def have(self, keys):
        self.asked.append(list(keys))
        return [(4, 2, 2) if k in self.held else None for k in keys]


def chat(*parts):
    return openai_to_messages({"messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, *parts]}]})


class PicturesById(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tok = ByteTokenizer()
        self.vision = KeyedVision(self.tmp.name)
        self.engine = RefEngine(self.tok, "ok", max_context=CTX)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=self.vision)

    def tearDown(self):
        self.svc.drop_embeddings()
        self.tmp.cleanup()

    def prepare(self, *parts):
        msgs, tools, kw = chat(*parts)
        ids, _, _ = self.svc.prepare(msgs, tools, kw, 8)
        path = self.svc.embeddings.path
        return ids, Path(path).read_bytes() if path else b""

    def test_the_uuid_part_is_kept_for_the_template(self):
        msgs, _, _ = chat({"type": "image_url", "image_url": None, "uuid": "frame-7"})
        self.assertEqual(msgs[0]["content"][1], {"type": "image", "source": "", "uuid": "frame-7"})

    def test_an_unknown_id_is_vllms_cache_miss(self):
        from serve.server import ImageCacheMiss
        self.prepare({"type": "image_url", "image_url": {"url": png(1)}, "uuid": "known"})
        self.svc.drop_embeddings()
        for gone in ({"image_url": None}, {}, {"image_url": {"url": ""}}, {"image_url": ""}):
            with self.subTest(part=gone):
                with self.assertRaises(ImageCacheMiss) as e:
                    self.prepare({"type": "image_url", "image_url": None, "uuid": "known"},
                                 {"type": "image_url", "uuid": "frame-1", **gone})
                self.assertEqual(str(e.exception), "Cache miss for image at index 1 but data is not provided.")
                self.svc.drop_embeddings()

    def test_a_known_id_needs_no_bytes_and_no_encode(self):
        self.prepare({"type": "image_url", "image_url": {"url": png(1)}, "uuid": "frame-1"})
        self.assertEqual(self.vision.encodes, 1)
        self.svc.drop_embeddings()
        ids, rows = self.prepare({"type": "image_url", "image_url": None, "uuid": "frame-1"})
        self.assertEqual(self.vision.encodes, 1)                     # the encoder's cache, by the id's key
        self.assertEqual(rows[:4], struct.pack("<i", 0x31455653))   # its rows (the engine does not hold it)

    def test_pictures_the_engine_holds_go_by_reference(self):
        self.prepare({"type": "image_url", "image_url": {"url": png(1)}, "uuid": "a"},
                     {"type": "image_url", "image_url": {"url": png(2)}, "uuid": "b"})
        key_a = self.svc.image_ids["a"]                              # a uuid's picture is keyed by the uuid
        self.engine.held = {key_a[:16]}
        self.svc.drop_embeddings()
        ids, rows = self.prepare({"type": "image_url", "image_url": None, "uuid": "a"},
                                 {"type": "image_url", "image_url": {"url": png(2)}, "uuid": "b"},
                                 {"type": "image_url", "image_url": {"url": png(3)}})
        self.assertEqual(self.vision.encodes, 3)                     # a and b once each, c now
        magic = struct.unpack_from("<i", rows, 0)[0]
        self.assertEqual(magic, SVR1)                                # a: a header, no rows
        self.assertEqual(struct.unpack_from("<5i", rows, 0), (SVR1, 4, 2, 2, KeyedVision.WIDTH))
        self.assertEqual(struct.unpack_from("<i", rows, 20)[0], 0x31455653)   # b: its rows
        self.assertEqual(len(self.svc.embeddings.keys), 3)
        self.assertEqual(self.svc.embeddings.keys[0], key_a[:16])
        self.assertEqual(ids.count(self.tok.encode("<|image_pad|>", parse_special=True)[0]), 12)
        refill = self.svc.embeddings.refill                          # NEED_IMAGES: every picture with its rows
        self.assertIsNotNone(refill)
        full = Path(refill()).read_bytes()
        self.assertEqual(full.count(struct.pack("<i", 0x31455653)), 3)
        self.assertEqual(self.vision.encodes, 3)

    def test_the_same_bytes_again_are_not_decoded_again(self):
        self.prepare({"type": "image_url", "image_url": {"url": png(5)}})
        self.svc.drop_embeddings()
        calls = []
        real = Vision.normalize
        try:
            Vision.normalize = staticmethod(lambda d: calls.append(1) or real(d))
            self.prepare({"type": "image_url", "image_url": {"url": png(5)}})
        finally:
            Vision.normalize = staticmethod(real)
        self.assertEqual(calls, [])
        self.assertEqual(self.vision.encodes, 1)


if __name__ == "__main__":
    unittest.main()


class CacheMissOverHttp(unittest.TestCase):
    """vLLM's body for the cache miss, which an API client looks for ("Cache miss")."""

    def test_body(self):
        import json
        import urllib.error
        import urllib.request
        from serve.server import serve
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as d:
            svc = Service(RefEngine(tok, "</think>\n\nok", max_context=CTX), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=KeyedVision(d))
            httpd = serve(svc, port=0)
            base = f"http://127.0.0.1:{httpd.server_address[1]}"

            def post(*parts):
                body = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "look"}, *parts]}]}
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status, json.load(r)
                except urllib.error.HTTPError as e:
                    with e:
                        return e.code, json.load(e)
            try:
                self.assertEqual(post({"type": "image_url", "image_url": {"url": png(1)}, "uuid": "f1"})[0], 200)
                self.assertEqual(post({"type": "image_url", "image_url": None, "uuid": "f1"})[0], 200)   # a hit
                code, body = post({"type": "image_url", "image_url": None, "uuid": "f1"},
                                  {"type": "image_url", "image_url": None, "uuid": "f2"})
                self.assertEqual(code, 400)
                self.assertEqual(body, {"error": {"message": "Cache miss for image at index 1 but data is not provided.",
                                                  "type": "BadRequestError", "param": None, "code": 400}})
                self.assertEqual(post({"type": "image_url", "image_url": None, "uuid": "f1"},
                                      {"type": "image_url", "image_url": {"url": png(2)}, "uuid": "f2"})[0], 200)
            finally:
                httpd.shutdown()
                httpd.server_close()


class SnapshotPoints(unittest.TestCase):
    """A client's own snapshot points (cache_control / prompt_cache_breakpoint) -> the engine's snap= cells."""

    def test_multiple_marks_keep_literal_spans_and_control_tokens(self):
        import copy
        import sys
        from unittest.mock import patch
        from serve.frontend import CACHE_MARK, mark_think_literals, unmark_think_literals
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_tokenizer as ST
        # Real BPE implementation with a tiny byte vocabulary, including the first
        # CONTROL tag (whose private marker once collided with CACHE_MARK).
        specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        tok = ST.Tokenizer([ST.BYTE_TO_UNICODE[b] for b in range(256)] + specials, [],
                           [1] * 256 + [3, 3, 3, 4, 4])
        svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertNotIn(CACHE_MARK, svc.literals.values())
        for count in (2, 3, 4, 5):
            with self.subTest(marks=count):
                parts = [{"type": "text", "text": f"блок {i} <think><|im_start|></think>",
                          **({"cache_control": {"type": "ephemeral"}} if i % 2 == 0 else
                             {"prompt_cache_breakpoint": {"mode": "explicit"}})} for i in range(count)]
                req = {"messages": [{"role": "user", "content": parts}]}
                plain_req = copy.deepcopy(req)
                for part in plain_req["messages"][0]["content"]:
                    part.pop("cache_control", None)
                    part.pop("prompt_cache_breakpoint", None)
                msgs, tools, kw = openai_to_messages(plain_req)
                expected = svc.encode_prompt(msgs, tools, kw)
                marked, marked_tools, _ = mark_think_literals(msgs, tools, svc.literals)
                clean_prompt, plain_spans = unmark_think_literals(svc.render_prompt(marked, marked_tools, kw), svc.literals)
                msgs, tools, kw = openai_to_messages(req)
                with patch.object(tok, "encode_marked", wraps=tok.encode_marked) as encode:
                    actual = svc.encode_prompt(msgs, tools, kw)
                self.assertEqual(actual, expected)
                self.assertEqual(encode.call_args.args[0], clean_prompt)
                self.assertEqual(encode.call_args.kwargs["plain"], plain_spans)
                snaps = svc.embeddings.snaps
                self.assertEqual(len(snaps), min(count, 4))
                # Every marker follows its block's final literal </think>.
                ends = [i + len("</think>") for i in range(len(clean_prompt)) if clean_prompt.startswith("</think>", i)]
                offsets = encode.call_args.args[1]
                self.assertEqual(offsets, ends[:count][-4:])
                self.assertTrue(all(0 < p < len(actual) for p in snaps))

    def test_private_marker_preserves_literal_unicode_in_both_apis(self):
        import copy
        from unittest.mock import patch
        from types import SimpleNamespace
        from serve.frontend import CACHE_MARK, CACHE_MARK_KEY, anthropic_to_messages
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        # Include both the old reserved codepoint and the first generated candidate.
        collision = CACHE_MARK + "a" * 32
        literal = "literal " + CACHE_MARK + " and " + collision + " <think>text</think>"
        for convert in (openai_to_messages, anthropic_to_messages):
            with self.subTest(api=convert.__name__):
                plain = {"messages": [{"role": "user", "content": [
                    {"type": "text", "text": literal}, {"type": "text", "text": " final"}]}]}
                msgs, tools, kw = convert(plain)
                self.assertNotIn(CACHE_MARK_KEY, kw)
                expected = svc.encode_prompt(msgs, tools, kw)
                self.assertIn(CACHE_MARK, tok.decode(expected))
                marked = copy.deepcopy(plain)
                for part in marked["messages"][0]["content"]:
                    part["cache_control"] = {"type": "ephemeral"}
                with patch("serve.frontend.uuid.uuid4", side_effect=[
                        SimpleNamespace(hex="a" * 32), SimpleNamespace(hex="b" * 32)]):
                    msgs, tools, kw = convert(marked)
                self.assertEqual(kw[CACHE_MARK_KEY], CACHE_MARK + "b" * 32)
                self.assertEqual(svc.encode_prompt(msgs, tools, kw), expected)
                # A client cannot forge the internal provenance via template kwargs.
                plain["chat_template_kwargs"] = {CACHE_MARK_KEY: CACHE_MARK}
                msgs, tools, kw = convert(plain)
                self.assertNotIn(CACHE_MARK_KEY, kw)
                self.assertEqual(svc.encode_prompt(msgs, tools, kw), expected)

    def test_marks_become_cells(self):
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        from serve.frontend import CACHE_MARK
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        req = {"messages": [{"role": "system", "content": [{"type": "text", "text": "rules " * 20,
                                                             "cache_control": {"type": "ephemeral"}}]},
                            {"role": "user", "content": [{"type": "text", "text": "history " * 10,
                                                           "prompt_cache_breakpoint": {"mode": "explicit"}},
                                                          {"type": "text", "text": "now?"}]}]}
        msgs, tools, kw = openai_to_messages(req)
        ids = svc.encode_prompt(msgs, tools, kw)
        plain = svc.encode_prompt(openai_to_messages({"messages": [
            {"role": "system", "content": "rules " * 20},
            {"role": "user", "content": [{"type": "text", "text": "history " * 10}, {"type": "text", "text": "now?"}]}]})[0],
            None, {})
        self.assertEqual(ids, plain)                                # the marks leave no trace in the prompt
        self.assertNotIn(CACHE_MARK, tok.decode(ids))
        self.assertIsNone(svc.embeddings.snaps)                     # (the byte tokenizer has no marked encode)
