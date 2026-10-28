#!/usr/bin/env python3
"""Importing the opt-in adapter must preserve the common server's defaults."""
from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root), str(root / 'tools'), str(root / 'tools/m31/serve')]
from serve import server as server
from serve import frontend
import strata_tokenizer

before = {
    'engine': server.StrataEngine,
    'service': server.Service,
    'detokenizer': server.Detokenizer,
    'template': frontend.ChatTemplate,
    'parser': frontend.OutputParser,
    'tokenizer': strata_tokenizer.Tokenizer,
    'vision': server.Vision,
    'content_parts': frontend._parts_of,
    'image_parts': frontend.IMAGE_PARTS,
    'anthropic_blocks': frontend.ANTHROPIC_BLOCKS,
    'end_token': server.IM_END,
    'thinking_end': server.THINK_END,
}
environment = dict(os.environ)
patterns = dict(strata_tokenizer.PRE_PATTERNS)
text = '<think>literal</think>\nРусский / English / 東京'
ids = server.ByteTokenizer().encode(text)
import serve_m31  # noqa: E402,F401

after = {
    'engine': server.StrataEngine,
    'service': server.Service,
    'detokenizer': server.Detokenizer,
    'template': frontend.ChatTemplate,
    'parser': frontend.OutputParser,
    'tokenizer': strata_tokenizer.Tokenizer,
    'vision': server.Vision,
    'content_parts': frontend._parts_of,
    'image_parts': frontend.IMAGE_PARTS,
    'anthropic_blocks': frontend.ANTHROPIC_BLOCKS,
    'end_token': server.IM_END,
    'thinking_end': server.THINK_END,
}
assert before == after, 'an adapter import changed the common server defaults'
assert environment == dict(os.environ), 'an adapter import changed process environment'
assert all(strata_tokenizer.PRE_PATTERNS[key] == value for key, value in patterns.items())
assert server.ByteTokenizer().encode(text) == ids
assert server.ByteTokenizer().decode(ids) == text
print('M31 import preserves common model classes, environment and token bytes')
