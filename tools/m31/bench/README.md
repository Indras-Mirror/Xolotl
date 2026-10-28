# Optional M31 checks

Run these clients in an isolated measurement window. HTTP clients use an
already running server; `prefix_speed.py` starts the explicitly named native
worker. They do not change system settings, reserve hardware, execute tools
or download models. Output directories must be new.

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
- `prefix_speed.py`: real live-KV append timing through the native line protocol.
  Defaults to a 1,024-token prefix and +100/+300 tokens, with one warmup and
  three measured repetitions per suffix. Every sample resets history with a
  two-token uncached request, primes the prefix, then appends with cache enabled.
  It requires the exact `RESUME 1024`, suffix `read` count and complete `PP`
  chunk trace. Only append wall time and `DONE prompt_ms` enter the summary;
  the first output head/sample remains in prompt time. Stores worker/input
  hashes, argv, all requests, warmups, protocol rows and stderr. Startup,
  response, input-write and shutdown deadlines are bounded.
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

## Live-prefix timing

`IDS` must name a little-endian int32 text-ID file containing at least 1,324
IDs. The public cache-observatory input and vocabulary-only tokenization recipe
in the main document produce a suitable file. Use identical IDs in each arm.
This example uses the measured single-5090/EPYC 9654 resource settings; smaller
budgets on different hardware are a changed recipe.

```sh
python tools/m31/bench/prefix_speed.py \
  --engine "$CANDIDATE" --model "$MODEL" --ids-file "$IDS" --out prefix-candidate \
  --prefix 1024 --suffixes 100,300 --repetitions 3 --backend cuda \
  --threads 80 --context 40000 --chunk 32768 --arena-gib 170 \
  --gpu-experts-gib 7 --register-arena 1 --register-limit-gib 176 --pack-threads 8
```

Run the same client with candidate/base/base/candidate workers and four new
output directories in one window. Hardware ownership belongs to the caller;
this client starts no controller and signals only its own child process. The
client also works with the PR1/base worker, without copying it into that tree.
Keep `receipt.json`, `requests.jsonl`, `stdin.log`, `stdout.log` and `stderr.log`
for every arm, including warmups and failed samples. A cold fallback or partial
reuse fails the run. This measures live cache appends; it does not compare their
numerics to a different cold-prefill chunk recipe.
