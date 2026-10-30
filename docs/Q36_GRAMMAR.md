# Qwen3.6 constrained output

`strata-q36` can constrain tokens with llama.cpp's GBNF sampler, using a
vocabulary-only load of the same GGUF. Configure `STRATA_Q36_GRAMMAR=ON` and
point `STRATA_GGML_DIR` at the complete llama.cpp source tree used for ggml.
With the option OFF, the engine does not advertise grammar support.
Constraints are selected by a request, not by a model-wide output format.

The vocabulary library uses a CPU-only ggml registry. It does not load another
model's weights or a second CUDA backend. The CUDA decoder and MTP remain q36's.

## Native protocol

```text
GEN 200 temperature=0.9 top_p=0.9 top_k=20 grammar=/absolute/request.gbnf 1,2,3
GENI 200 grammar=/absolute/request.gbnf grammar_after=THINK_END_ID /absolute/image.sve 1,2,3
```

`grammar` names a UTF-8 GBNF file with a `root` rule. The grammar is compiled
before prefill; a failure returns `ERR grammar: ...` without executing the
request. `grammar_after` delays constraints until after that token is emitted.
Without it, the grammar applies to the first generated token. Paths cannot
contain whitespace. The server creates its own files; HTTP clients send GBNF
text, never a native filename.

The engine advertises `INFO grammar=1`. For constrained sampling, it masks
forbidden tokens before penalties, top-k, temperature, min-p and top-p. The
top-128 device candidates are sufficient only when the retained cut lies above
every excluded raw logit; otherwise it copies and masks the full logits row.
Greedy chooses the highest allowed penalized logit.
If the device's raw argmax is allowed and no penalties apply, it is already the
exact constrained greedy result and no top-128 pass is needed. MTP drafts use
the same shortcut whenever their raw argmax is allowed; their proposal remains
the greedy allowed token. Sampling verification always computes its target
distribution with the requested penalties and truncation.

Full masks are memoized by the native grammar's complete pushdown state and
partial UTF-8 state, so different free-text values can reuse an identical mask.
The small state accessor compiles the unchanged pinned llama.cpp sampler source
in one wrapper translation unit; it does not modify the vendor tree or parse
GBNF. Sampler clones keep independent mutable state. The cache holds at most
eight grammars, with 8192 states and 32 MiB of masks per grammar. A narrow cached
mask can use its complete allowed-ID list after the full logits copy. This is
an exact representation of the full mask, not another top-k cutoff.

MTP drafts are deterministic greedy choices among the grammar-allowed top-128
draft logits. If that list has no allowed token, the unconstrained draft is
verified normally. Each window clones the grammar state; only accepted tokens
advance its verifier state. A deterministic draft `d` is accepted with
probability `p(d)`; rejection samples the target distribution with `d` removed.
Greedy accepts a draft only when it matches the allowed argmax. At a completed
terminal branch only `<|im_end|>` can end the answer.

## Chat Completions

These fields select one constraint:

| API style | Fields |
| --- | --- |
| OpenAI | `response_format.type=json_object` or `json_schema` with `json_schema.schema` |
| vLLM | `structured_outputs.json`, `.grammar`, `.choice`, or `.json_object=true` |
| Legacy vLLM | `guided_json`, `guided_grammar`, `guided_choice` |
| llama.cpp | `grammar`, `json_schema` |

`structured_outputs.regex` and `guided_regex` return 400 with a “not supported”
message. Structured output with tools or MCP also returns 400. Conflicting
constraints return 400. Identical grammar texts in top-level `grammar` and
`structured_outputs.grammar` are accepted together. GBNF escapes such as
`\x0A` pass unchanged to llama.cpp.

Set `"structured_compact": true` in the serve config to remove optional JSON
whitespace. A request can override this default with
`structured_outputs.disable_any_whitespace` (a boolean). This flag does not
rewrite raw GBNF or remove spaces inside JSON strings.

When thinking is enabled, serve sends `grammar_after` for `</think>`. JSON
answers still pass the existing final schema validation. Engines without
`INFO grammar=1` keep the previous prompt-and-validate `response_format` path.

Streaming requests with `stream_options.include_usage=true` end with a separate
usage chunk whose `choices` is empty, followed by `[DONE]`. This also applies
to unconstrained requests. Without that flag the previous final chunk is kept.

## Checks

`python -m unittest serve.test_structured` checks HTTP normalization, native
protocol keys, lazy activation, compaction, conflicts and the final usage chunk
with MockEngine. The other serve tests require no model or GPU.

Build the optional `q36-grammar-check` target and run it with `MODEL SCENE_GBNF`
to check real vocabulary masks, cloning, lazy activation, EOS and exact fast
versus full distributions on CPU.

`tools/q36_grammar_bench.py` records native validity, MTP distribution comparisons
(TV, split-sample noise and chi-square p-values), ABBA decode cost and a
pre-change greedy regression check. `tools/q36_grammar_http_check.py` exercises
the real server, optionally with image input. Both accept explicit engine,
model, tokenizer and evidence-directory paths. They generate synthetic GBNF
fixtures unless `--grammars` supplies a custom directory. Reserve the GPU
through your scheduler before running either script.

For decode cost, `--perf-sampling --perf-repeats 104` measures greedy and the
example application temperatures, with 50 requests per path after four warmups.
The free-text values vary between ABBA groups, with each path receiving every
case twice. The evidence includes warmup responses, binary and grammar hashes,
individual responses, and the fixed prompts used for distribution checks.

The measured warm decode overhead is 0.15–5.23%. The first sampled request
builds native full-vocabulary masks and costs an additional 152–232 ms on the
tested application prompts. The <=10% target is therefore met for the warm path only;
new grammar states can incur further mask-building cost. See the complete
[hardware validation report](Q36_GRAMMAR_VALIDATION.md), including cold requests
and both the original distribution test and its independent replication.

`tools/q36_grammar_baseline.py` builds the original q36 translation unit from
a caller-selected `--baseline-ref` using the current build's flags and libraries. It writes only to its
evidence directory and does not create or modify another worktree. Build this
baseline before any timed runs.
