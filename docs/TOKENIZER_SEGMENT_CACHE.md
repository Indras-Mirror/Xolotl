# Reusing tokenized plain segments

`STRATA_TOKENIZER_SEGMENT_CACHE_MIB=32` opts the Python tokenizer into a
bounded in-memory cache of exact plain segments. The default is `0` (off).
Set the variable on the server process, before its tokenizer is constructed.
The cache belongs to that tokenizer instance; it is not shared between models
or saved to disk.

Special-token splitting and protected literal spans still run normally. A
complete plain segment between those boundaries can reuse its token IDs when
the entire string matches. An appended or changed suffix is tokenized normally;
the cache never assumes that tokenization of a character prefix is reusable.
This can reduce preparation time when requests resend long unchanged message
or image-separated text segments. It does not reuse model KV or image embeddings.

The existing `PromptEncoder` continues to reuse shared special-token prefixes.
This cache also works when that prefix changes or a conversation has left
`PromptEncoder`'s recent entries: complete unchanged later segments can hit.
It preserves `encode_marked` resume points and `encode_at_offsets` snapshot
frontiers as separate APIs. Neither API infers a BPE boundary from a character
prefix.

Snapshot offsets use a separate exact key containing the segment and relative
character offsets. The cached token frontiers are the ones found by the normal
pre-tokenizer on the miss. Caller mark IDs and absolute prompt positions are
rebound on every hit. Plain and marked entries share one byte budget and one
64-entry LRU. Segments shorter than 256 characters, oversized entries, and more
than 16 marks in one segment bypass memoization. Token IDs and stored frontiers
are immutable; every encode returns a fresh list.

The budget conservatively charges retained text, IDs, mark metadata and LRU
entry overhead; the small fixed cache/lock object is outside that payload
budget. The existing per-piece and token-byte caches are separate and are not
covered by this additional cache's budget. Lookup and eviction locks do not cover tokenization. Requests with no
repeated complete segment will receive little or no benefit.

For diagnostics, `STRATA_TOKENIZER_SEGMENT_CACHE_TRACE=1` prints cumulative
numeric cache counters after an encode: budget, charged bytes, entry count,
hits and misses. It prints no text, token IDs or image identifiers. Trace
counters belong to the tokenizer, not to an individual concurrent request.

Run `python -m unittest tools.test_tokenizer_segment_cache
tools.test_tokenizer_plain_spans tools.test_strata_tokenizer` for CPU correctness
checks. End-to-end latency requires separate warm requests with changing tails,
images and concurrent callers; tokenizer timings alone are not TTFT.
