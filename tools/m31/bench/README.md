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
  in the TSV and full raw responses. Optional `--nonce-seed STRING` gives
  identical request bodies across matched ABBA arms; omitting it retains random
  uuid4 nonces. The seed is recorded only when explicitly supplied.
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

## Matched ABBA HTTP bodies

Use the same corpus, tokenizer, resources, repetition count and nonce seed in
all arms. Request index 0 is the warmup, followed by 4K/16K/32K. The nonce is the
hex of `UUID(bytes=SHA256(f"{seed}-{index}".encode()).digest()[:16])`, without
rewriting version or variant bits. The measured PR3 window used this recipe:

```sh
python tools/m31/bench/speed.py --url "$URL" --tokenizer "$TOKENIZER" \
  --corpus "$CORPUS" --repetitions 1 --nonce-seed m31-pr3-abba-fixed \
  --out speed-arm-A1
```

Repeat with base/base/candidate servers and three new output directories in
the same window. The public flag reproduces the original local driver's fixed
nonce sequence; it leaves the default unseeded client and timing denominators
unchanged. Preserve every raw response and warmup as usual.

## Record the actual HTTP chunk recipe

`trace_server.py` runs the actual M31 common-server shim from a chosen checkout.
It forwards the original stdout records unchanged and records only
INFO/READY/PP/RESUME/DONE/ERR. It does not log T tokens, change model arithmetic,
reserve hardware or alter the caller's environment. Use the same wrapper and
speed client for both branches, with the caller owning the GPU throughout.

```sh
mkdir trace-A1
python tools/m31/bench/trace_server.py --source "$CHECKOUT" \
  --trace trace-A1/protocol.jsonl -- --engine strata --config "$CONFIG" \
  --tokenizer "$TOKENIZER" --host 127.0.0.1 --port 8080
# In another terminal, after /health is ready:
python tools/m31/bench/speed.py --url http://127.0.0.1:8080 \
  --corpus "$CORPUS" --tokenizer "$TOKENIZER" --repetitions 1 \
  --nonce-seed m31-pr3-abba-fixed --out trace-A1/speed
python tools/m31/bench/trace_chunks.py trace-A1/protocol.jsonl \
  --requests trace-A1/speed/raw.jsonl --out trace-A1/chunks.json
```

Use new output directories for all ABBABAAB arms. `trace_chunks.py` requires the
one-repetition warmup/4K/16K/32K recipe, complete PP coverage, zero reuse, exact
output budgets and one engine PID per arm. Hidden restarts or incomplete traces
fail. Keep actual chunks and responses even when planner partitions differ.
The exact historical speed corpus requires its archived input; its SHA and
reproduction limit are in the main document. A different corpus is a new test.

## Compare complete prefix-state heads

`compare_prefix_heads.py` checks the existing `m31_cache_model_check` outputs at
1024+100 and 1024+300. It requires all nine complete 200064-F32 heads, finite
values, correct raw top-1 metadata, exact forward/decode/replay recipes and raw
byte equality with the baseline. It independently checks within-worker rewind,
decode-rewind, restore, repeated cold prefill and workspace release invariants.
Whole-versus-split histories remain separate. There is no tolerance option or
last-row subset that can replace a missing head.

```sh
python tools/m31/bench/compare_prefix_heads.py cache-A-1124 cache-B-1124 \
  --length 1124 --out cache-1124-comparison.json
python tools/m31/bench/compare_prefix_heads.py cache-A-1324 cache-B-1324 \
  --length 1324 --out cache-1324-comparison.json
```

The main document contains the four real-model checker commands and the exact
public input generator. Model-free regression cases for both tools run as part
of `test_bench.py` and the existing m31 benchmark ctest group. They start no
model, encoder or GPU process.
