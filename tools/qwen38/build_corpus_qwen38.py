#!/usr/bin/env python3
"""Run tools/v41/build_corpus.py with OUR model's tokenizer instead of DeepSeek's.

Why this exists
---------------
`tools/v41/build_corpus.py` hardcodes MODEL (env V41_MODEL) and does
`AutoTokenizer.from_pretrained(MODEL)`.  Pointing that at the Qwen3.8-Flash-Next FP8
checkpoint fails before any corpus work starts: AutoTokenizer builds an AutoConfig first, and
transformers does not know `model_type: qwen4_exp` ("The checkpoint you are trying to load has
model type `qwen4_exp` but Transformers does not recognize this architecture").  The model class
is irrelevant to corpus building - only `tok.encode(text, add_special_tokens=False)` is used
(build_corpus.py:337,349,359) - so we build the tokenizer straight from tokenizer.json with the
`tokenizers` library and inject it.

This wrapper patches AutoTokenizer.from_pretrained, then runs the unmodified builder.  Nothing in
tools/v41 is edited: the DeepSeek calibration path keeps working exactly as before.

  python3 tools/qwen38/build_corpus_qwen38.py --max-tokens 2000000
  # same args as build_corpus.py; V41_WORK still selects the output dir
"""
import os
import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).resolve().parent
XOLOTL = HERE.parent.parent
V41 = XOLOTL / "tools" / "v41"

# the checkpoint we are quantising; its tokenizer.json is the one the model actually uses
MODEL_DIR = pathlib.Path(os.environ.get(
    "QWEN38_MODEL_DIR", "/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8")).expanduser()

# default the work dir to somewhere with room, unless the caller set one
os.environ.setdefault("V41_WORK", "/media/mal/NVME1TB/qwen38")

from tokenizers import Tokenizer  # noqa: E402

_tok_path = MODEL_DIR / "tokenizer.json"
if not _tok_path.is_file():
    raise SystemExit(f"build_corpus_qwen38: no tokenizer at {_tok_path}")


class _EncodeResult(list):
    """What build_corpus.py needs from an encode() call: something len()-able."""


class _TokenizerShim:
    """The two knobs build_corpus.py actually touches."""

    def __init__(self, path: pathlib.Path):
        self._t = Tokenizer.from_file(str(path))

    def encode(self, text, add_special_tokens=False):
        # add_special_tokens is a no-op here: HF tokenizer.json post-processors for this model
        # add nothing unless asked, and build_corpus.py always passes False.
        return _EncodeResult(self._t.encode(text).ids)


def _from_pretrained(*args, **kwargs):
    # build_corpus.py makes exactly one tokenizer call, for the calibration corpus, so returning
    # the shim unconditionally is correct and avoids guessing which path form it was handed
    # (it is handed DeepSeek's MODEL, not ours, and transformers chokes before we see it).
    return _TokenizerShim(_tok_path)


import transformers  # noqa: E402

_ORIG = transformers.AutoTokenizer.from_pretrained
transformers.AutoTokenizer.from_pretrained = staticmethod(_from_pretrained)

_probe = Tokenizer.from_file(str(_tok_path)).encode("def foo(): return 1")
print(f"build_corpus_qwen38: tokenizer ready, vocab={_probe.ids and len(_probe.ids)} ids "
      f"for a sample line", file=sys.stderr)

sys.argv[0] = str(V41 / "build_corpus.py")
runpy.run_path(str(V41 / "build_corpus.py"), run_name="__main__")
