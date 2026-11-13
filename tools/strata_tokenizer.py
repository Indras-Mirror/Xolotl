"""tools/strata_tokenizer.py - the tokenizer, as the reference implementation of record.

WHY THIS EXISTS AND WHERE IT SITS.  The GGUF carries the whole tokenizer in metadata: `tokenizer.ggml.model`
= `gpt2`, 248,320 `tokens`, 247,587 `merges`, `token_type`, `pre` = `qwen35`, and a Jinja `chat_template`.
None of that needs the 35 GB of weights, so it is extracted into the pack's `tokenizer/` directory
(docs/pack-format.md §6) and this module is what reads it back.  A C++ port follows and is tested against
this one.

THE DECISIVE PROPERTY IS THE ROUND TRIP.  Byte-level BPE maps each BYTE to a printable unicode character so
that any UTF-8 input is representable with no UNK token.  Every failure mode of that mapping - the wrong
offset for a byte, a merge applied in the wrong order, a pre-tokenizer split that drops a character - still
produces plausible token ids.  `decode(encode(s)) == s` is the check that sees them, and it is exact because
the byte layer is lossless.  It is asserted over a corpus chosen to hit the boundaries: multi-byte UTF-8,
emoji (4-byte), combining marks, whitespace runs, and C0 control bytes.
"""
from __future__ import annotations

import bisect
import json
import os
import pathlib
import sys
import threading
from collections import OrderedDict

import regex

# ------------------------------------------------------------------ byte <-> unicode (GPT-2 byte-level BPE)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte encoder: 256 bytes -> 256 printable characters, reversibly.

    Bytes 33..126, 161..172 and 174..255 map to themselves; the remaining 68 (space, newline, the C0
    controls and the high range that would be invisible) are shifted into 256+n so that no byte is
    unprintable.  The shift is the whole trick and getting its ORDER wrong is invisible in the output ids.
    """
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# The `qwen35` pre-tokenizer, transcribed from the ORACLE rather than from the family resemblance:
# `.ref/llama.cpp/src/llama-vocab.cpp` L396, `case LLAMA_VOCAB_PRE_TYPE_QWEN35`.  The commented-out line
# above it is the `tokenizer.json` original, and llama.cpp's active version differs from it - it spells the
# contraction classes out instead of using `(?i:...)`, which is behaviourally the same.
#
# THE `\p{M}` IS THE WHOLE POINT OF THIS BEING TRANSCRIBED.  The QWEN3 pattern two cases earlier (L389) is
# the same shape with `\p{L}` where this has `[\p{L}\p{M}]` and without `\p{M}` in the punctuation negation.
# Writing the QWEN3 pattern for `qwen35` looks right, round-trips perfectly, and is wrong: a combining mark
# is `\p{M}`, so without it the mark gets swallowed into the following punctuation run and a token like `_j`
# never forms.  It cost 3 strings out of 1875 and only `tokenize_oracle_check` could see it.
QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# The `joyai-llm` pre-tokenizer (DeepSeek-V4-Flash and the DeepSeek3/Hunyuan family), transcribed from
# `.ref/llama.cpp src/llama-vocab.cpp` L320-326 (`case LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM`).  UNLIKE qwen35 this is
# THREE regexes, and llama.cpp APPLIES THEM SEQUENTIALLY: each one further splits the pieces the previous one
# left (`unicode_regex_split` at `unicode.cpp:1291-1367` feeds the accumulating `bpe_offsets` into the next).
# The order is load-bearing.  `\p{N}{1,3}` must cut digit runs first (so `01234` is `012|34`); then
# `[一-龥぀-ゟ゠-ヿ]+` pulls Han / kana runs out of a larger `\p{L}+` (so `abc漢字def` is `abc|漢字|def`); the
# last pattern is the ordinary word splitter.  A single alternation of the three looks equivalent on ASCII and
# is not: it keeps `abc漢字def` whole.
JOYAI_LLM_PATTERNS = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    (r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"
     r"|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
     r"| ?[\p{P}\p{S}]+[\r\n]*"
     r"|\s*[\r\n]+"
     r"|\s+(?!\S)"
     r"|\s+"),
]

# The `lfm2` pre-tokenizer (Liquid AI's LFM2 / LFM3 vocabulary, local: LFM3-Omni): llama.cpp maps it to
# LLAMA_VOCAB_PRE_TYPE_LLAMA3 (src/llama-vocab.cpp, upstream f3586e6: the `tokenizer_pre == "lfm2"` case) with
# ignore_merges = true - a pre-token piece that is itself in the vocabulary is that one token, not merged (see _bpe).
LLAMA3_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# The `llama4` pre-tokenizer (Llama 4 and Meta's Muse Glimmer / Muse Spark vocabulary, 202,048 tokens): llama.cpp maps it
# to LLAMA_VOCAB_PRE_TYPE_GPT4O (src/llama-vocab.cpp, the `tokenizer_pre == "llama4"` case, clean_spaces off).  This is
# the tokenizer.json original, which the `regex` module reads as it is; llama.cpp's own copy spells the case classes as
# lookaheads because std::regex has no \p{Lu}.
GPT4O_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# pre type -> the pattern list llama.cpp applies, in order.  The `pre` string in GGUF metadata selects it.
PRE_PATTERNS = {
    "qwen35": [QWEN35_PATTERN],
    "joyai-llm": JOYAI_LLM_PATTERNS,
    "lfm2": [LLAMA3_PATTERN],
    "llama4": [GPT4O_PATTERN],
}
IGNORE_MERGES = {"lfm2"}   # llama.cpp's ignore_merges: a whole piece found in the vocabulary is taken as it is


# The `deepseek-v3` pre-tokenizer (DeepSeek-V3 / V4.1; llama.cpp LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM, the same regex list as
# JoyAI and Hunyuan-dense).  UNLIKE the single-pattern families it is a SEQUENCE: each pattern splits the pieces the
# one before it left (a span a pattern does not match stays a piece of its own), so digits (runs of 1..3) and CJK
# runs are cut out before the general word / punctuation / whitespace pattern sees the rest.
DEEPSEEK3_PATTERNS = (
    r"\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
    r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)


def _sequence_findall(patterns, text: str) -> list[str]:
    """llama.cpp's unicode_regex_split: the pieces of `text` after each pattern in turn splits every piece into its
    matches and the unmatched spans between them."""
    pieces = [text]
    for pat in patterns:
        nxt = []
        for piece in pieces:
            pos = 0
            for m in pat.finditer(piece):
                if m.start() > pos:
                    nxt.append(piece[pos:m.start()])
                nxt.append(m.group())
                pos = m.end()
            if pos < len(piece):
                nxt.append(piece[pos:])
        pieces = nxt
    return pieces


class _PlainSegmentCache:
    """Bounded, tokenizer-local memoization AFTER special-token splitting.

    An exact plain string is the complete input to the existing BPE routine.
    Never infer a reusable token prefix from a character prefix: a newly appended
    character can change pre-tokenizer and BPE boundaries. Values are immutable;
    callers get their own output list, and concurrent encodes may safely evict
    or compute the same entry. The lock covers metadata only, never tokenization.
    """
    MIN_CHARS = 256
    MAX_ENTRIES = 64

    def __init__(self, budget: int):
        if budget < 0:
            raise ValueError("tokenizer segment cache budget must be nonnegative")
        self.budget = budget
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, text, mark_offsets=None):
        if not self.budget or len(text) < self.MIN_CHARS or (mark_offsets is not None and len(mark_offsets) > 16):
            return None
        key = text if mark_offsets is None else (text, tuple(mark_offsets))
        with self.lock:
            item = self.entries.get(key)
            if item is None:
                self.misses += 1
                return None
            self.hits += 1
            self.entries.move_to_end(key)
            return item[0] if mark_offsets is None else (item[0], item[2])

    def put(self, text, ids, mark_offsets=None, mark_frontiers=None):
        if not self.budget or len(text) < self.MIN_CHARS or (mark_offsets is not None and len(mark_offsets) > 16):
            return
        offsets = None if mark_offsets is None else tuple(mark_offsets)
        frontiers = None if mark_frontiers is None else tuple(mark_frontiers)
        key = text if offsets is None else (text, offsets)
        value = tuple(ids)
        # Conservative retained-entry accounting: charge each integer even when
        # it is shared with vocab/piece caches, plus room for the LRU nodes/table.
        # The small fixed cache object/lock are outside this payload budget.
        # IDs index a Python vocabulary list, so charging a full platform-sized
        # integer per ID also bounds their payload without walking a long miss
        # for sys.getsizeof on every token.
        charge = sys.getsizeof(text) + sys.getsizeof(value) + len(value) * sys.getsizeof(sys.maxsize) + 256
        if offsets is not None:
            # Mark offsets are relative to this segment, not a caller's absolute
            # prompt position or its mark IDs. Preserve the exact pre-tokenizer
            # frontiers found on the miss; no approximate offset-to-token mapping.
            charge += sys.getsizeof(key) + sys.getsizeof(offsets) + sum(map(sys.getsizeof, offsets))
            charge += sys.getsizeof(frontiers) + sum(
                sys.getsizeof(point) + (sum(map(sys.getsizeof, point)) if point is not None else 0)
                for point in frontiers)
        if charge > self.budget:
            return
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous is not None:
                self.bytes -= previous[1]
            while self.entries and (self.bytes + charge > self.budget or len(self.entries) >= self.MAX_ENTRIES):
                _, old = self.entries.popitem(last=False)
                self.bytes -= old[1]
            self.entries[key] = (value, charge, frontiers)
            self.bytes += charge

    def stats(self):
        with self.lock:
            return {"budget_bytes": self.budget, "charged_bytes": self.bytes,
                    "entries": len(self.entries), "hits": self.hits, "misses": self.misses}


class Tokenizer:
    def __init__(self, tokens: list[str], merges: list[str], token_types: list[int] | None = None,
                 pre: str = "qwen35", special_ids: dict[str, int] | None = None,
                 *, segment_cache_bytes: int | None = None):
        self.tokens = tokens
        self.pre = pre
        self.token_types = token_types
        self.special_ids = special_ids or {}
        self.ids = {t: i for i, t in enumerate(tokens)}
        if len(self.ids) != len(tokens):
            raise ValueError("vocabulary has duplicate tokens: %d entries, %d unique"
                             % (len(tokens), len(self.ids)))
        # Merge rules as (left, right) -> rank.  A merge list that is not a total order over its pairs, or one
        # naming a token that is not in the vocabulary, would make BPE silently produce different ids than the
        # model was trained with, so both are checked here rather than discovered as a bad answer later.
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) != 2:
                raise ValueError("merge %d is not a pair: %r" % (i, m))
            if parts[0] not in self.ids or parts[1] not in self.ids:
                raise ValueError("merge %d names a token outside the vocabulary: %r" % (i, m))
            self.ranks[(parts[0], parts[1])] = i
        # xolotl + upstream: the fork's per-`pre` pattern lists (`_pre_res`, applied as a sequence by `_pre_split`) take
        # upstream's DeepSeek-V3 sequence too; `_split` is upstream's name for the same pieces.
        pats = DEEPSEEK3_PATTERNS if str(pre).startswith("deepseek") else PRE_PATTERNS.get(pre)
        if pats is None:
            raise ValueError("unknown pre-tokenizer %r (known: %s, deepseek*)"
                             % (pre, ", ".join(sorted(PRE_PATTERNS))))
        self._pre_res = [regex.compile(p) for p in pats]
        self._split = self._pre_split
        self.ignore_merges = pre in IGNORE_MERGES

        # The literals matched directly instead of being run through BPE.  GGUF token types: 3 = CONTROL,
        # 4 = USER_DEFINED.  The two classes behave DIFFERENTLY and llama.cpp's own tokenizer settled which:
        #
        #   * type 4 (USER_DEFINED: `<think>`, `<tool_call>`, `<tool_response>`, ...) is matched ALWAYS,
        #     with or without parse_special.
        #   * type 3 (CONTROL: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, ...) is matched ONLY when
        #     parse_special is set.
        #
        # Measured, not assumed: with parse_special=False the oracle still emitted `<tool_response>` as one
        # token and mine emitted four, and the three disagreements were exactly the type-4 cases while every
        # type-3 case agreed.  Treating both classes alike costs 3 strings in 1229 and silently changes a chat
        # prompt, because `<|im_end|>` decomposed into ordinary pieces is not the token the model expects.
        self.special_tokens: dict[str, int] = {}
        if token_types:
            for i, ty in enumerate(token_types):
                if ty in (3, 4):
                    self.special_tokens[tokens[i]] = i
        always = [t for t, i in self.special_tokens.items() if token_types and token_types[i] == 4]
        self.control_tokens = [t for t, i in self.special_tokens.items() if token_types and token_types[i] == 3]
        # Longest literal first, or `<|im_end|>` could match a shorter prefix of itself.  `regex.escape` so a
        # token containing regex metacharacters (several do: `<|`, `[`, `(`) is matched literally.
        self._always_re = self._alt(always)
        self._special_re = self._alt(list(self.special_tokens))
        # Initialize once, rather than allocating a default dict and looking it up
        # through __dict__.setdefault on each encoded segment or decoded token.
        self._piece_ids: dict[str, list[int]] = {}
        self._bytes_cache: dict[int, bytes] = {}
        # How far a special-token match can look ahead of where it starts (PromptEncoder's margin).
        self.max_special_len = max((len(t) for t in self.special_tokens), default=1)

        if segment_cache_bytes is None:
            try:
                segment_cache_bytes = int(os.environ.get("STRATA_TOKENIZER_SEGMENT_CACHE_MIB", "0")) * 1024 * 1024
            except ValueError as exc:
                raise ValueError("STRATA_TOKENIZER_SEGMENT_CACHE_MIB must be a nonnegative integer") from exc
        self._segment_ids = _PlainSegmentCache(segment_cache_bytes)
        self._segment_trace = os.environ.get("STRATA_TOKENIZER_SEGMENT_CACHE_TRACE") == "1"

    def _trace_segment_cache(self):
        if self._segment_trace:
            sys.stderr.write("STRATA_TOKENIZER_SEGMENT_CACHE " +
                             json.dumps(self._segment_ids.stats(), separators=(",", ":")) + "\n")

    @staticmethod
    def _alt(literals: list[str]):
        if not literals:
            return None
        return regex.compile("|".join(regex.escape(s) for s in sorted(literals, key=len, reverse=True)))

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_gguf(cls, path) -> "Tokenizer":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from gguf_reader import GGUFFile
        md = GGUFFile(pathlib.Path(path)).metadata
        need = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
        missing = [k for k in need if k not in md]
        if missing:
            raise ValueError("GGUF is missing %s; it does not carry a tokenizer" % ", ".join(missing))
        model = md.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError("expected a byte-level BPE tokenizer (model 'gpt2'), got %r" % model)
        special = {k: int(md[k]) for k in md
                   if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        return cls(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"]),
                   list(md.get("tokenizer.ggml.token_type") or []) or None,
                   md.get("tokenizer.ggml.pre", "qwen35"), special)

    # -------------------------------------------------------------- the algorithm
    def _bpe(self, word: str) -> list[str]:
        """Merge `word` (already byte-mapped) by LOWEST RANK first, repeatedly - not left to right.

        Applying merges in list order rather than rank order is the classic BPE bug: it produces a different
        segmentation and a plausible token count.

        A word longer than HEAP_MIN symbols (a long CJK run, a minified blob: one pre-token piece of thousands of
        bytes) goes to _bpe_heap, which applies the same merges in the same order without rescanning every pair
        after each merge (#268: that rescan is O(n^2) and took seconds on an 8K-character CJK run).
        """
        if getattr(self, "ignore_merges", False) and word in self.ids:   # llama.cpp ignore_merges (lfm2)
            return [word]
        if len(word) > self.HEAP_MIN:
            return self._bpe_heap(word)
        # `self` is read ONCE, here.  #1385: an interpreter (CPython 3.14.4) was seen handing this frame an int
        # for `self` partway through the scan (`'int' object has no attribute 'ranks'`).  Nothing in this class
        # can do that (no cache, decorator, slots or callback; a thread hammering test cannot make it happen),
        # so the scan below works on a local and no longer re-reads `self` once per symbol pair.
        ranks_get = self.ranks.get
        parts = list(word)
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = ranks_get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return parts

    HEAP_MIN = 64       # words up to this many symbols keep the scan above, unchanged

    def _bpe_heap(self, word: str) -> list[str]:
        """_bpe's result in O(n log n): the symbols as a linked list, every adjacent pair with a rank in a heap
        keyed (rank, position).  Popping the lowest rank and then the lowest position is exactly the order the
        scan picks (its strict `<` keeps the leftmost of equal ranks), and a symbol's position is where it starts
        in the word, which only its left neighbour's merge can change - so a popped pair whose two symbols are
        no longer the ones it was pushed with is stale and skipped."""
        import heapq
        parts = list(word)
        n = len(parts)
        nxt = list(range(1, n)) + [-1]
        prv = list(range(-1, n - 1))
        ranks = self.ranks
        heap = []
        for i in range(n - 1):
            r = ranks.get((parts[i], parts[i + 1]))
            if r is not None:
                heap.append((r, i, parts[i], parts[i + 1]))
        heapq.heapify(heap)
        while heap:
            r, i, left, right = heapq.heappop(heap)
            j = nxt[i]
            if parts[i] != left or j < 0 or parts[j] != right:
                continue                                # stale: one of its symbols has merged since
            parts[i] = left + right
            parts[j] = None                             # j is gone: i takes its place in the list
            k = nxt[j]
            nxt[i] = k
            if k >= 0:
                prv[k] = i
                r2 = ranks.get((parts[i], parts[k]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, i, parts[i], parts[k]))
            p = prv[i]
            if p >= 0:
                r2 = ranks.get((parts[p], parts[i]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, p, parts[p], parts[i]))
        return [s for s in parts if s is not None]

    def _pre_split(self, text: str) -> list[str]:
        """The pre-tokenizer pieces of `text`: each pattern in `_pre_res` splits the pieces the patterns before
        it left, exactly as llama.cpp's `unicode_regex_split` refines `bpe_offsets` one regex at a time."""
        spans = [(0, len(text))]
        for pat in self._pre_res:
            nxt: list[tuple[int, int]] = []
            for a, b in spans:
                sub = text[a:b]
                last = 0
                for m in pat.finditer(sub):
                    if m.start() > last:
                        nxt.append((a + last, a + m.start()))
                    if m.end() > m.start():
                        nxt.append((a + m.start(), a + m.end()))
                    last = m.end()
                if last < len(sub):
                    nxt.append((a + last, b))
            spans = nxt
        return [text[a:b] for a, b in spans]

    PIECE_CACHE_MAX = 200_000    # pre-tokenizer pieces remembered (an agent resends its whole history every turn)

    def _encode_plain(self, text: str, marks=None, base: int = 0, out_marks=None) -> list[int]:
        if marks:   # the marks' token positions: a mark inside a piece goes to the piece's start (a token frontier)
            return self._encode_plain_marked(text, marks, base, out_marks)
        if not self._segment_ids.budget or len(text) < self._segment_ids.MIN_CHARS:
            return self._encode_plain_uncached(text)
        memo = self._segment_ids.get(text)
        if memo is not None:
            return list(memo)
        out = self._encode_plain_uncached(text)
        self._segment_ids.put(text, out)
        return out

    def _encode_plain_uncached(self, text: str) -> list[int]:
        """The existing BPE/piece-cache path, also used for ineligible segments."""
        out: list[int] = []
        # A piece's ids depend on the piece alone, so repeated pieces (most of a resent conversation) are looked up
        # instead of merged again.  The ids are the ones _bpe gives: this only skips the work.
        cache = self._piece_ids
        # xolotl: _pre_split, not the dense lineage's `self._re.findall` - the fork replaced `_re` with the
        # per-`pre` pattern list `_pre_res` (joyai-llm = 3 sequential regexes); for qwen35's single pattern the two
        # give the same pieces (the pattern covers every character, so there are no unmatched gaps).
        for piece in self._pre_split(text):
            got = cache.get(piece)
            if got is None:
                mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
                got = []
                for tok in self._bpe(mapped):
                    i = self.ids.get(tok)
                    if i is None:
                        raise KeyError("BPE produced a token outside the vocabulary: %r" % tok)
                    got.append(i)
                if len(cache) < self.PIECE_CACHE_MAX:
                    cache[piece] = got
            out.extend(got)
        return out

    def _encode_plain_marked_uncached(self, text: str, marks, base: int, out_marks) -> list[int]:
        # Keep the legacy piece-first algorithm for OFF/short/many-mark input.
        # No offset tuple, frontier array or per-piece segment-cache calls.
        out: list[int] = []
        a = 0
        for piece in self._pre_split(text):
            b = a + len(piece)
            for k, c in marks:
                if base + a <= c < base + b and k not in out_marks:
                    out_marks[k] = (len(out), base + a)
            out.extend(self._encode_plain_uncached(piece))
            a = b
        return out

    def _encode_plain_marked(self, text: str, marks, base: int, out_marks) -> list[int]:
        if (not self._segment_ids.budget or len(text) < self._segment_ids.MIN_CHARS or
                not isinstance(marks, (list, tuple)) or len(marks) > 16):
            return self._encode_plain_marked_uncached(text, marks, base, out_marks)
        offsets = tuple(c - base for _, c in marks)
        memo = self._segment_ids.get(text, offsets)
        if memo is not None:
            ids, frontiers = memo
            # The reference walks pieces before marks. With duplicated caller
            # IDs, the earliest piece wins even if input offsets are unsorted.
            ordered = sorted((point[1], i, point) for i, point in enumerate(frontiers) if point is not None)
            for _, i, point in ordered:
                k = marks[i][0]
                if k not in out_marks:
                    out_marks[k] = (point[0], base + point[1])
            return list(ids)
        out: list[int] = []
        frontiers = [None] * len(marks)
        a = 0
        for piece in self._pre_split(text):
            b = a + len(piece)
            for i, ((k, _), c) in enumerate(zip(marks, offsets)):
                if a <= c < b:
                    frontiers[i] = (len(out), a)
                    if k not in out_marks:
                        out_marks[k] = (len(out), base + a)
            out.extend(self._encode_plain_uncached(piece))
            a = b
        self._segment_ids.put(text, out, offsets, frontiers)
        return out

    def encode_at_offsets(self, text: str, marks: list[int], parse_special: bool = False, plain=()) -> tuple[list[int], list[int]]:
        # xolotl (the client's cache-marker offsets); upstream's encode_marked below is a different call (PromptEncoder)
        """encode(), and for each character offset in `marks` the number of tokens before it - the offset taken back to
        the start of its pre-tokenizer piece (or special token) when it falls inside one, so that every position is a
        frontier of the encoding itself (pieces never merge across)."""
        pat = self._special_re if parse_special else self._always_re
        pending = sorted((c, k) for k, c in enumerate(marks))
        got: dict[int, int] = {}
        out: list[int] = []
        def plain_part(t, start):
            local: dict[int, tuple[int, int]] = {}
            inside = [(k, c) for c, k in pending if start <= c < start + len(t)]
            ids = self._encode_plain_marked(t, inside, start, local) if inside else self._encode_plain(t)
            for k, (n, _) in local.items():
                got.setdefault(k, len(out) + n)
            out.extend(ids)
        pos = 0
        matches = [] if pat is None else list(pat.finditer(text))
        for m in matches:
            start = m.start()
            if plain and any(a <= start < b for a, b in plain):
                continue
            if start > pos:
                plain_part(text[pos:start], pos)
            for c, k in pending:
                if start <= c < m.end():
                    got.setdefault(k, len(out))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            plain_part(text[pos:], pos)
        n = len(out)
        self._trace_segment_cache()
        return out, [got.get(k, n) for k in range(len(marks))]

    def _encode_matching(self, text: str, pat, plain=(), marks=None) -> list[int]:
        """Encode `text`, emitting any literal `pat` matches as single tokens and BPE-ing the rest.

        The split happens on the RAW text, before the byte mapping, because a special token's string is a
        literal to match rather than bytes to decompose.  Everything between the matches is tokenized
        normally - which is why a near-miss like `<|im_star` still costs ordinary tokens.  A match that starts
        inside one of the `plain` (start, end) spans is left to the text around it (#537).

        `marks`, when given, receives (end of the match in `text`, ids so far) for every match: the points where
        the encoding of a longer text with the same beginning can resume (PromptEncoder).
        """
        if pat is None:
            return self._encode_plain(text)
        out: list[int] = []
        pos = 0
        # Matches arrive in text order. For reusable span sequences, consume each
        # span once instead of scanning the entire list for every literal match.
        # Sorting a copy handles unsorted/overlapping input without mutating it.
        # Other iterables keep the original consumption semantics below.
        span_sequence = isinstance(plain, (list, tuple)) and bool(plain)
        if span_sequence:
            spans = iter(sorted(plain))
            upcoming = next(spans, None)
            plain_end = 0
        for m in pat.finditer(text):
            start = m.start()
            if span_sequence:
                while upcoming is not None and upcoming[0] <= start:
                    plain_end = max(plain_end, upcoming[1])
                    upcoming = next(spans, None)
                if start < plain_end:
                    continue
            elif plain and any(a <= start < b for a, b in plain):
                continue
            if start > pos:
                out.extend(self._encode_plain(text[pos:start]))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
            if marks is not None:
                marks.append((pos, len(out)))
        if pos < len(text):
            out.extend(self._encode_plain(text[pos:]))
        return out

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        """Tokenize `text`.

        `parse_special` controls only the type-3 CONTROL literals such as `<|im_end|>`; the type-4
        USER_DEFINED ones such as `<think>` are matched either way.  See the note in `__init__`.
        `plain`: (start, end) spans of `text` that are ordinary text even where they spell a literal - a
        `</think>` quoted in a message (#537) - and are tokenized with the text around them.
        """
        ids = self._encode_matching(text, self._special_re if parse_special else self._always_re, plain)
        self._trace_segment_cache()
        return ids

    def encode_marked(self, text: str, parse_special: bool = False, plain=()) -> tuple[list[int], list[tuple[int, int]]]:
        """encode() and its resume points: (end offset in `text`, ids so far) after every special-token match."""
        marks: list[tuple[int, int]] = []
        ids = self._encode_matching(text, self._special_re if parse_special else self._always_re, plain, marks)
        self._trace_segment_cache()
        return ids, marks

    def token_bytes(self, i: int) -> bytes:
        """The raw bytes of one token (a multi-byte character can be split across tokens)."""
        cache = self._bytes_cache
        b = cache.get(i)
        if b is None:
            if i < 0 or i >= len(self.tokens):
                raise IndexError("token id %d is outside the vocabulary (%d)" % (i, len(self.tokens)))
            if self.token_types and i < len(self.token_types) and self.token_types[i] in (3, 4):
                # a special token's string is the literal text (DeepSeek's `<｜User｜>`, `｜DSML｜` hold characters outside
                # the byte alphabet); for Qwen's ASCII specials it is the same bytes the alphabet would give
                b = cache[i] = self.tokens[i].encode("utf-8")
                return b
            raw = bytearray()
            for ch in self.tokens[i]:
                v = UNICODE_TO_BYTE.get(ch)
                if v is None:
                    raise KeyError("token %d contains a character outside the byte alphabet: %r" % (i, ch))
                raw.append(v)
            b = cache[i] = bytes(raw)
        return b

    def decode(self, ids: list[int], errors: str = "replace") -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors=errors)


# ------------------------------------------------------------------ incremental prompts (#567)
def common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix, by slices (C speed) rather than a loop over characters."""
    n = min(len(a), len(b))
    lo, step = 0, 4096
    while lo < n:                                       # whole chunks first, doubling
        hi = min(n, lo + step)
        if a[lo:hi] != b[lo:hi]:
            break
        lo, step = hi, min(step * 2, 1 << 20)
    else:
        return n
    while hi - lo > 1:                                  # the first difference is in [lo, hi): bisect it
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def plain_agree_until(a, b, limit: int) -> int:
    """The first position below `limit` where the `plain` spans (#537) of two texts differ - a position inside a
    span in one and not in the other - or `limit`.  Spans are (start, end), sorted and disjoint."""
    def clip(spans):
        return [(x, min(y, limit)) for x, y in spans if x < limit and y > x]
    ca, cb = clip(a), clip(b)
    for (xa, ya), (xb, yb) in zip(ca, cb):
        if (xa, ya) != (xb, yb):
            return min(xa, xb) if xa != xb else min(ya, yb)
    if len(ca) != len(cb):
        return (ca if len(ca) > len(cb) else cb)[min(len(ca), len(cb))][0]
    return limit


class PromptEncoder:
    """`tok.encode(text, parse_special=True, plain=...)` for chat prompts, reusing the ids of an earlier prompt.

    A chat client sends the whole conversation every turn, and the rendered prompt of turn n+1 starts with most of
    turn n's.  Running BPE over all of it again costs hundreds of milliseconds at 100K tokens, even when the engine
    reuses the whole prefix.  So the ids up to a SPECIAL-TOKEN BOUNDARY both prompts share are taken as they were,
    and only the rest is encoded.

    WHY THAT IS EXACT.  Encoding splits the text at special-token matches and BPE-encodes each stretch between two
    matches on its own (`_encode_matching`): nothing crosses a match.  Whether a match is taken at position q
    depends only on text[q : q + max_special_len] and on whether q lies inside a `plain` span (#537: a </think>
    quoted in a message is text).  So if the two texts agree up to L and their plain spans agree below L, every
    decision the scan makes at a position before c = the end of some match, with c + max_special_len - 1 <= L, is
    the same for both - the same matches, the same stretches, the same ids up to c - and from c the scan starts
    afresh, exactly as it does on text[c:] with the spans shifted by c.  Hence encode(new) == ids_old[:ids at c] +
    encode(new[c:]).  The margin matters only for a vocabulary where a literal overlaps another's end; it costs
    re-encoding one short stretch.

    A few recent prompts are kept (one per conversation: a prompt replaces the one it continues, that is, one whose
    text it reuses for more than half; a prompt that shares only the template's head with every entry is a new
    entry), so a second client does not evict the first.  `encode(..., store=False)` reads the kept prompts without
    writing one (a token count is not a conversation's turn).  The tokenizer needs `encode_marked` and
    `max_special_len`.
    """

    def __init__(self, tok, keep: int = 4):
        self.tok, self.keep = tok, keep
        # (text, plain spans, ids, mark ends, mark counts)
        self.entries: list[tuple[str, tuple, list[int], list[int], list[int]]] = []
        self.lock = threading.Lock()
        self.last_reused = 0                            # characters taken from an earlier prompt (for tests)

    def encode(self, text: str, plain=(), store: bool = True) -> list[int]:
        plain = tuple(sorted(plain))
        with self.lock:
            entries = self.entries                      # replaced, never changed in place
        margin = self.tok.max_special_len - 1
        src, cut_k = None, -1
        for e in entries:
            same = plain_agree_until(e[1], plain, common_prefix_len(e[0], text))
            k = bisect.bisect_right(e[3], same - margin) - 1   # the last boundary c with c <= same - margin
            if k >= 0 and (src is None or e[3][k] > src[3][cut_k]):
                src, cut_k = e, k
        if src is None:
            ids, marks = self.tok.encode_marked(text, parse_special=True, plain=plain)
            ends, counts = [m[0] for m in marks], [m[1] for m in marks]
            self.last_reused = 0
        else:
            c, n = src[3][cut_k], src[4][cut_k]
            rest = tuple((max(x - c, 0), y - c) for x, y in plain if y > c)
            tail, marks = self.tok.encode_marked(text[c:], parse_special=True, plain=rest)
            ids = src[2][:n] + tail
            ends = src[3][:cut_k + 1] + [c + m[0] for m in marks]
            counts = src[4][:cut_k + 1] + [n + m[1] for m in marks]
            self.last_reused = c
        if not store:
            return ids
        # every rendered prompt starts with the template's header, so sharing a special boundary does not make two
        # prompts one conversation: only the entry whose text is mostly reused is replaced
        old = src if src is not None and 2 * src[3][cut_k] > len(src[0]) else None
        with self.lock:
            self.entries = [(text, plain, ids, ends, counts)] + [e for e in self.entries if e is not old][:self.keep - 1]
        return list(ids)

    def forget(self, ids: list[int]) -> None:
        """The prompt with these ids was refused: it is not kept."""
        with self.lock:
            self.entries = [e for e in self.entries if e[2] != ids]


# ------------------------------------------------------------------ the pack's tokenizer/ directory
def extract(gguf_path, out_dir) -> dict:
    """Write the tokenizer into `<out_dir>/tokenizer/` so the engine never opens the weight shards for it."""
    out = pathlib.Path(out_dir) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer.from_gguf(gguf_path)
    cfg = {
        "model": "gpt2",
        "pre": tk.pre,
        "vocab_size": len(tk.tokens),
        "n_merges": len(tk.ranks),
        "special_ids": tk.special_ids,
        "add_bos_token": False,
        # The pattern is SHIPPED, not recomputed by the reader: it is transcribed from llama.cpp for the
        # declared `pre` type, and a C++ port that re-derived it would be free to get `\p{M}` wrong again.
        "pre_pattern": QWEN35_PATTERN,
        "pre_pattern_source": ".ref/llama.cpp src/llama-vocab.cpp L396 (LLAMA_VOCAB_PRE_TYPE_QWEN35)",
    }
    (out / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join("%s %s" % k for k, _ in
                                              sorted(tk.ranks.items(), key=lambda kv: kv[1])), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(tk.token_types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    # the model's own chat template: fine-tunes change it (Swift 1.5 differs from Qwen3.8-Flash-Next's)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    tpl = GGUFFile(pathlib.Path(gguf_path)).metadata.get("tokenizer.chat_template")
    if tpl:
        (out / "chat_template.jinja").write_text(tpl, encoding="utf-8", newline="\n")
    return cfg


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="round-trip a corpus and report")
    args = ap.parse_args()
    cfg = extract(args.gguf, args.out)
    print("tokenizer/: vocab %d, merges %d, pre %s, specials %s"
          % (cfg["vocab_size"], cfg["n_merges"], cfg["pre"], cfg["special_ids"]))
    if args.check:
        tk = Tokenizer.from_gguf(args.gguf)
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
                  "def f(x):\n\treturn x  # comment\n", "\u4f60\u597d\uff0c\u4e16\u754c", "\u0645\u0631\u062d\u0628\u0627",
                  "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "e\u0301\u0301 combining", "\x00\x01\x7f control",
                  "\u00a0non-breaking\u00a0space", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n", "\u2028\u2029",
                  "x" * 5000]
        bad = 0
        for s in corpus:
            ids = tk.encode(s)
            back = tk.decode(ids)
            if back != s:
                print("  *** ROUND TRIP FAILED *** %r -> %d ids -> %r" % (s[:40], len(ids), back[:40]))
                bad += 1
        print("round trip: %d strings, %d failed" % (len(corpus), bad))
        return 0 if bad == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
