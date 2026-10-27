# Optional M31 checks

Run these clients against an already running engine/server in an isolated
measurement window. They do not change system settings, reserve hardware,
execute tools or download models. Output directories must be new.

- `quality.py`: first 100 ordered GSM8K main test rows, or the frozen 20 tool
  prompts plus three continuations. Temperature 1, top-p 0.95, top-k 40,
  min-p 0.05, maximum 6,000 and model-default thinking. Keeps full requests/replies
  and never executes tools. Optional `--seed UINT64` supports declared replications;
  omitting it leaves the request unseeded.
- `speed.py`: portable `om_bench` HTTP recipe. Warmup, then 4K/16K/32K repeated;
  greedy, 256 generated tokens, low reasoning, no prompt cache. Supply the corpus
  explicitly. Pass `--tokenizer EXPORT_DIR` for the common server (which has no
  tokenizer HTTP routes); reference runs can use their HTTP tokenizer. The server
  must return separate prefill and decode timings. Both servers are normalized
  to `prompt_n / prompt_seconds` and `(generated - 1) / decode_seconds`; the first
  output head and sample belong to prompt time. Original reported rates remain
  in the TSV and full raw responses.
- `make_prompts.py`: fixed 20-case numerical suite, or long corpus fixtures.
  `--suite long-public` needs no corpus and generates RU/code at 16K and 32K
  (four cases, 2,048 varied passages per source). `--suite long --corpus FILE`
  preserves the historical corpus recipe. The oracle's vocabulary-only mode
  makes `.i32` files for both engines. `--suite cache-observatory` publishes the
  exact 257/1324-token input used to observe whole-versus-split sensitivity;
  its manifest records the corresponding 256/1024-token reuse boundaries.
- `replay_oracle.py`: runs the separately built pinned oracle sequentially using
  the candidate's actual chunk plan. An optional bounded-capacity mode verifies
  contiguous KV positions, padded active keys and unchanged trained RoPE.
- `compare_logits.py`: streams F32 logits using NumPy. Checks token IDs, positions,
  row counts, actual chunks, placement and final-layer crop policy before metrics.
  It reports aggregate numerical acceptance and byte identity separately.

[The main document](../../../docs/MINIMAX_M31.md) has the pinned revisions, build
and run commands, numerical contract and results. The completed PR1 collection
is recorded in [pr1-text.json](receipts/pr1-text.json), with every quality trial
and every speed repetition. PR1's initial tools trial (17/23) and the reference's
seed-1 trial (18/23) failed the unchanged 19/23 threshold. Seeded trials do not
replace those outcomes. The final decode rates were about 14–15% below historical
v8; no whole-model cause was established.

Keep source/build/model identities with each result. A new corpus, tokenizer,
chunk plan or placement is a changed recipe and must be named in a report.
The published measurements do not establish performance on untested hardware.

`tool_fixtures.py` contains the exact historical multilingual prompt data. Paths
and service names in that data are inert strings presented to the model; the
client never accesses them or executes a returned call.
