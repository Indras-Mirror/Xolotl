# Qwen4-Max (OpenRouter API) on Gauntlet v3-small — 2026-11-16

The reference row for the **Qwen4 weights** we will download, ablate and quantise locally. Until a local Qwen4
quant exists, this API run is the score that quant has to be measured against, on the identical task set, turns,
thinking budget and harness.

Model: `qwen/qwen4-max` (OpenRouter, served by Alibaba; 1M context, thinking summaries).
Raw artifacts: `gauntlet-greedy.json`, `gauntlet-t1.json` (each carries the full per-step log),
`throughput.json` in this directory.

## Result

| arm | solved | SCORE /6 | mean solve turn | truncated | completion tokens | model time | e2e tok/s | cost |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy (temp 0) | **5/6** | **5.50** | 5.0 | 0 | 47,241 | 831 s | 56.8 | **$0.34** |
| temp 1.0 | **5/6** | **5.50** | 4.8 | 0 | 42,493 | 817 s | 52.0 | **$0.33** |

The two arms agree on both the score and the shape: the same five tasks pass, `03-raytracer` misses at the same
0.50 (scene B), and no step is truncated. Temperature 1.0 gets to `04-log-needle` in 5 turns instead of 9 and to
`09-dst-scheduler` in 6 instead of 4 — the same solves, different path.

`03-raytracer` is the only miss, and it is narrow — scene A renders correctly, scene B is **3.1 % of pixels off by
more than 2** against a 1 % threshold (mean error 0.32, limit 0.5). Same failure mode as every other model that
misses it: the scene-B geometry, not a collapse.

| task | what it tests | solved at | steps | tokens | secs | cost |
| --- | --- | :---: | ---: | ---: | ---: | ---: |
| 01-ledger-rootcause | three-file money bug, contract in docstrings | turn 4 | 4 | 7,521 | 138 | $0.048 |
| 03-raytracer | full raytracer, pixel-exact vs reference | — (0.50) | 5 | 4,790 | 81 | $0.035 |
| 04-log-needle | 39K-token access log, rate-limit audit | turn 9 | 9 | 11,459 | 204 | $0.100 |
| 06-ttl-cache | LRU + TTL + stats + thread safety | turn 3 | 3 | 7,155 | 128 | $0.037 |
| 09-dst-scheduler | DST-correct scheduler from a docstring | turn 4 | 4 | 10,621 | 178 | $0.075 |
| 10-flaky-suite | find the order-dependent flaky test | turn 5 | 5 | 5,695 | 102 | $0.049 |

Zero truncated steps (no thinking turn hit the 32,768-token budget), and no integrity events (nothing tried to
write into the grading names). 30 model requests total, 135,174 prompt tokens, peak prompt 18,377 tokens.

**Thinking:** 26,801 reasoning tokens = **57 % of all output** (temp 1.0: 22,581 = 53 %). Qwen4-Max thinks a lot
but finishes: the tasks it solved after turn 2 each needed one long thinking turn (`04` used 20K+ characters of
reasoning before its winning write).

Per-task numbers for the temp-1.0 arm: 01 @5 (6,158 tok), 03 fail 0.50 (9,937), 04 @5 (2,173), 06 @3 (5,229),
09 @6 (12,633), 10 @5 (6,363).

## Throughput

Measured over the wire with `tools/gauntlet/throughput-api.py` (streaming; TTFT = prefill, first-to-last content delta
= decode). These are end-to-end numbers — network and provider queueing included — so they are a **lower bound** on
the provider's engine speed and are not comparable to a local decode number measured from the engine's own timings.

| probe | prompt tokens | TTFT (prefill) | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| ~7K prompt, 256 out | 6,802 | 2.25 s | 3,026 | 56.1 |
| ~17K prompt, 256 out | 17,484 | 2.08 s | 8,415 | 59.5 |

In the real tasks (30 requests, all with thinking on) the aggregate was **56.8 output tok/s** and 163 prompt
tok/s over model wall time; per-request median 16.2 s, p90 68.3 s, max 123 s.

## Against the other API rows

Same 6 tasks, 10 turns, 32K thinking budget, greedy. API scores are the values published in the top-level
README on 8 Nov; the local rows are the models as served (not comparable point-for-point — they are there to show
where the API row sits).

| model | API SCORE /6 | local quant /6 |
| --- | ---: | ---: |
| GLM-5.3-Flash | 5.87 | — |
| **Qwen4-Max (this run)** | **5.50** | — (weights not released) |
| Qwen3.8-27B Swift-1.5 XQ 14 GB (ours) | 4.47 | **6.00** |
| DeepSeek-V4.1-Flash-1017 V4 | 4.49 | 4.99 |
| Qwen3.8-27B XQ 16 GB (ours) | 4.47 | 4.97 |
| Qwen3.8-Flash-Next XQ 45 GB | 3.90 | 4.49 |

Read it as: Qwen4-Max is the **second-strongest API on this set**, behind GLM-5.3-Flash, and it is the only API
that solved `04-log-needle` and `09-dst-scheduler` in one greedy pass. Qwen4-Max at temp 0 and the local 27B at
6.00 on served sampling are not the same measurement — the local number is the target, not a beat.

## What the local Qwen4 quant needs to reproduce

The comparison is only valid if the local arm runs the same recipe. After the download/ablate/quantise pipeline
(`xq/notes/QWEN4_PREP_2026-11-09.md`) produces a GGUF:

```bash
# serve it, then:
TASKEVAL_THINK_BUDGET=32768 GAUNTLET_TURNS=10 \
  python3 tools/gauntlet/gauntlet.py run http://127.0.0.1:<port> out/qwen4-local-r1 --only 01,03,04,06,09,10
```

and score the two together:

```bash
python3 tools/gauntlet/gauntlet.py score out/gauntlet-v3sm-qwen4max-or out/qwen4-local-r1
```

The API arm itself is reproducible with:

```bash
bash tools/gauntlet/run-api-qwen4max-v3sm.sh greedy   # temp 0
bash tools/gauntlet/run-api-qwen4max-v3sm.sh t1       # recommended sampling
```

## Caveats

- One greedy run, as every other row in the README. The harness decodes at temperature 0, so a rerun repeats the
  same tokens; `TASKEVAL_TEMPERATURE=served TASKEVAL_SEED=N` gives sampled runs to average.
- The temp-1.0 arm sets temperature only. Qwen's card also recommends `top_p 0.95 / top_k 20`; the harness does not
  expose either, so that arm is "temperature 1.0 at the provider's default top_p", not the full recommended block.
- The API publishes no engine timings, so there is no decode/prefill split for the task run itself — only the
  streamed probe above.
- Pricing is the OpenRouter list rate ($1.50/M in, $4.50/M out); the $0.34 is the metered cost of this run,
  including cached-prefix discounts.
