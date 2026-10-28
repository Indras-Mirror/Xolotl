# Qwen3.8-Flash-Next (`qwen4exp`) — our own abliterated, XQ-allocated quant

Goal: one artifact. Official Qwen base, our abliteration baked in, tier map chosen by the XQ
allocator from measured error instead of a depth heuristic. No more stacking four third-party
quants on top of each other.

Status: **download running, pipeline scaffolded, nothing quantised yet.**

---

## 0. What is different from DeepSeek-V4.1, and why

| | DeepSeek-V4.1 (XQ as built) | Qwen3.8-Flash-Next (this) |
|---|---|---|
| Routed experts | 15,360 | 24,576 (48 x 512) |
| Expert size | 35.4M params (5120x2304 SwiGLU) | 4.9M params (2560x640 SwiGLU) |
| Expert params | 543.6B | 120.8B |
| + PLE n-gram table | none | **51.2B** (separate tensor, separate shard set) |
| Quantiser | GPTQ -> PTQ1_0 / Q2_0 / Q3_K | llama-quantize + imatrix -> IQ2/IQ3/Q2_0/IQ4_NL |
| Allocation granularity | per expert | **per layer, per projection** (see §4) |
| Engine path | `deepseek4` | `qwen4exp` |

Two hard constraints, both measured, that shape everything below:

1. **The ternary tier does not exist on this path.** `include/strata/kernels/cpu/ptq1_0.hpp:1`
   scopes PTQ1_0 to "DeepSeek-V4.1 mixed-tier quant", and `iq_kernels.hpp:137` gates the PROM
   grouped kernels to PTQ1_0/Q2_0. The `qwen4exp` engine reads its experts through
   `native_expert.hpp`, i.e. ggml's own types. Proof from a shipped file: the SC117 IQ3_XXS build
   uses IQ2/IQ3 gate+up and Q2_0/IQ4_NL down — no ternary anywhere.
2. **GPTQ cannot make IQ2/IQ3.** Those are codebook/lattice formats produced by
   `llama-quantize` with an importance matrix. GPTQ produces scalar grids (Q2_0, Q3_K, Q4_K...).
   So "our same XQ full sweep" splits: the *allocation* carries over intact, the *GPTQ step* only
   applies to scalar tiers.

---

## 1. Files here

| file | purpose |
|---|---|
| `fetch_fp8.sh` | resumable, hash-verified fetch of the FP8 weights (136.0 GB) |
| `fetch_ple.sh` | the IQ4_NL PLE table (28.80 GB), queued to start after the weights |
| `fp8-manifest.json` | 113 files: size + sha256 (LFS) / sha1 (git blobs) |
| `skipped-ple-shards.json` | the 31 pure-PLE shards not downloaded |
| `check_coverage.py` | **guard**: proves every tensor is fetched or in a pure-PLE shard |
| `emit_tensor_types.py` | allocator map -> `llama-quantize --tensor-type-file` |
| `filter_index.py` | writes an index that references only the downloaded shards |
| `convert_ple_deferred.py` | converter wrapper that tolerates the deferred PLE table |
| `ggml_qshim.cpp` | C ABI over `ggml_quantize_chunk` — the only way to measure a real IQ-type error |
| `screen.py` | per-expert, per-tier held-out error -> the allocator's `stats/L*.json` |
| `dump_moe_all.patch` | Xolotl patch: dump every layer's MoE input, every chunk |
| `fp8-index.json` | the HF safetensors index, kept for the coverage check |
| `../v41/v41_alloc.py` | the XQ allocator; `--arch qwen4exp` selects this model's ladder |

### Why the coverage guard exists

The first version of the manifest classified a shard as "PLE-only" if it held a `.ple.` tensor and
no experts. `model-00037-of-00131` satisfies that test but also holds a whole layer's
`block_inject_weight`, `conv1d`, `hc_norm`, `in_proj_*`, `out_proj` — dropping it yields a model
that loads and then produces nonsense. A shard is only skippable when **every** tensor in it is
PLE. `check_coverage.py` enforces that and is meant to be run before any conversion.

```
python3 tools/qwen38/check_coverage.py
# tensors total 152089  fetched 151965  skipped(PLE) 124
# expert tensors 150528  all fetched: True
# OK: every tensor is either fetched or in a pure-PLE shard
```

---

## 2. Disk plan (Team CX2, ntfs3 — the fast-serve filesystem)

| step | bytes |
|---|---|
| FP8 weights, excluding PLE shards | 136.0 GB |
| IQ4_NL PLE table | 28.8 GB |
| BF16/F16 intermediate GGUF (converter output) | ~200 GB **not yet budgeted** |
| final quant | ~45-55 GB |

Team CX2 has 214 GB free. The two downloads fit (164.8 GB, leaving ~49 GB) but the conversion
intermediate does not. Options, in order of preference:

1. Convert and quantise in one stream if the converter supports it (avoids the intermediate).
2. Write the intermediate to NVME1TB (117 GB free, exFAT — fine for a scratch file that is read
   once, never served).
3. Free ~150 GB on Team CX2 first.

**Serve from ntfs3 or ext4, never exFAT** — measured 2026-10-18: expert reads 0.96 GB/s from
exFAT vs 5.1-5.3 GB/s from ntfs3.

---

## 3. Pipeline

```
  FP8 safetensors (136 GB, verified, hash-identical to the official manifest)
        |
        |  convert_hf_to_gguf.py   (llama.cpp-qwen4exp, conversion/qwen4exp.py)
        v
  model-Q8_0.gguf  (~145 GB)   <- Q8_0, NOT F16: F16 is ~272 GB and does not fit on any drive here
        |
        |  *** ABLITERATION EDIT — MUST BE HERE, NOT LATER (see 3.0) ***
        |  reject edit: 145 targets, the adapter in
        |  ~/AI/quetza-workspace/flashnext-ablit + alpha_preflight.py
        v
  model-Q8_0-ablated.gguf      <- the model that calibration must measure
        |
        |  llama-imatrix  <-- OUR corpus (code / agentic / chat mix)
        v
  imatrix.gguf
        |
        |  engine forward pass, STRATA_DUMP_MOE_ALL  (routed activations per layer)
        v
  dumps/  ->  screen.py (per-expert, per-tier output error)  ->  stats/L*.json
        |
        |  v41_alloc.py --arch qwen4exp   (seconds)
        v
  alloc.json  --emit_tensor_types.py-->  tensor_types.txt        (one type per layer: LOSSY)
             \-- v41_gguf.py mixed-layout writer --> per-expert types + PTQ1_0  (REQUIRED)
        |
        |  llama-quantize --imatrix ... --tensor-type-file ...   OR the mixed-layout writer
        v
  Qwen3.8-Flash-Next-XQ.gguf  +  PLE-IQ4_NL.gguf   (serve with --ple-gguf)
```

### 3.0 Ordering rule: abliteration runs BEFORE calibration

Verbatim from `private/docs/V41_ABLITERATION_PIPELINE.md`:

> **It must run before the sensitivity/calibration pass.** Even a dense-only edit changes the
> hidden states that feed the MoE and therefore the router inputs and the per-expert error
> measurements. Running calibration on the base and applying the edit later would fit the RCO
> tier map to a different model. […] **manifest first, then calibration**.

The 2026-10-27/28 run violated this: `qwen38-pipeline.sh` had **no abliteration stage at all**
(`grep -i abli` → nothing), and steps 3-5 read the pristine
`/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8` (verified byte-identical to
`fp8-manifest.json`). 48 of the 145 edit targets are `ffn_down_exps` — the routed experts — so the
per-expert error measurements described a model we do not ship. Stopped at STEP5 31/48 on
2026-10-28 05:15; those 31 stats files are kept in `stats-pristine-control/` as the control arm
for the sensitivity test in §6.1, **not** for allocation.

Two consequences that are not obvious:

1. **The edit changes the tier map, not just the weights.** `screen.py` fits the imatrix from the
   actual weights and scores output error on the actual forward pass; both move when the edit lands.
2. **Edited tensors cannot go ternary or low Q2_0.** §5.4 measured Q2_0 holding **1.2%** of the edit
   (`signal%`) vs IQ4_NL 17.3% and Q8_0 94.6% — a low-bit tier on an *edited* tensor erases the
   abliteration. That is a hard floor on the tier ladder for the 145 edited tensors, and it is why
   the ladder and the edit have to be decided together rather than in sequence.

### 3.1 Convert (two steps, both required)

**Why Q8_0 and not F16.** MEASURED on `layers.1.mlp.experts.0.down_proj` (real F8_E4M3 weights,
dequantised through their BF16 `weight_scale_inv` blocks):

| comparison | relative error |
|---|---:|
| FP8 weights -> Q8_0 | **0.552%** |
| FP8 weights -> BF16 | 0.153% |
| what FP8 already discarded | **~2.55%** (E4M3 median step 8.83%, RMS err ~2.55%) |

Q8_0's error is 4.6x smaller than the quantisation error FP8 itself already imposed, and 14x
smaller than the lightest XQ tier (IQ4_NL 7.8%). F16 would be more faithful in a vacuum but costs
~272 GB — no drive here holds it — and buys 0.4pp that the first XQ tier erases 14 times over.
Cross-check: `alpha_preflight.py` measured Q8_0 base_err 0.57% independently on layers 1/3/20.

**Why not the FP8 table.** The checkpoint is block-scaled FP8 (`weight` + `weight_scale_inv` per
128x128 block); the converter dequantises to float and re-quantises, so the intermediate's exact
precision is set by `--outtype`, not by the source.

The 31 pure-PLE shards are not on disk, which breaks the converter twice. Both breakages are known
and handled without editing the vendored llama.cpp tree:

```
# 1. index that only references shards we have
python3 tools/qwen38/filter_index.py --model-dir "/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8"

# 2. convert with the table deferred
QWEN_PLE_DEFERRED=1 python3 tools/qwen38/convert_ple_deferred.py \
  "/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8" \
  --outfile /some/where/model-Q8_0.gguf --outtype q8_0
```

**Why step 1.** The converter builds its shard list *from the index* (`conversion/base.py:246`,
`part_names` from `weight_map.values()`) and opens every one of them, so an unmodified index fails
on files that are not there. `filter_index.py` drops exactly the tensors in skipped shards and
verifies every referenced shard exists. Because it drops whole shards' worth and keeps the two
mixed shards intact, the converter's own consistency check (`base.py:282`, index vs parts
symmetric difference) still passes — measured 0 on the real index. `--restore` puts the original
back.

**Why step 2.** `_place_ple_shard` returns `[]` until it has all `split_ngram_parts` shards, so a
partial set simply produces no table tensor — but `prepare_tensors` then raises
"got 4 PLE embedding shards, expected 128". `convert_ple_deferred.py` replaces that single method
with a version that swallows only that specific error, and only when `QWEN_PLE_DEFERRED=1`;
without the env var it is a no-op. Nothing in `~/AI/llama.cpp-qwen4exp` is modified.

The metadata still gets what the engine needs: the hash constants and row width come from the
mixed shards we do download (`model-00005` carries `layer_multipliers`, `key_proj`, norms;
`model-00037` carries `ngram_heads_offsets`, `ngram_heads_vocab_sizes`, `value_proj`), so
`embedding_length_per_layer_input` = 160 is still written. The table itself arrives at serve time
via `--ple-gguf` — the same split `tools/nvfp4_convert.py` documents.

**Not yet verified:** that the engine accepts a GGUF with PLE keys and no
`per_layer_token_embd` tensor. The first load with `--ple-gguf` is the test.

### 3.2 imatrix

```
llama-imatrix -m model-F16.gguf -f corpus.txt -o imatrix.gguf --chunks N -t 8
```

The corpus is the XQ value-add and should be the same domain mix the DS4 calibration used:
roughly 40% code, 30% agentic/tool, 30% chat/rest, with a held-out slice. `tools/v41/build_corpus.py`
already builds this mix from local sources and can be pointed at a different output.

**MEASURED 2026-10-27 — size the ubatch, not just the context.** Each 512-token batch routes to
~all 512 experts per layer (top-10 of 512 over 512 tokens covers ~99.99%), so the expert working
set of *one physical batch* is the whole ~128 GB expert set against a ~87 GB page cache. The I/O
cost is therefore per **ubatch**, not per token: 512-token ubatches over 396k tokens read ~39 TB;
8192-token ubatches read ~6 TB. It was the difference between a measured 2.6 min/chunk (33 h for
the run) and ~2 min/chunk for 48 chunks.

Disk speed matters just as much. `dd iflag=direct` on this box: Crucial1TB (ntfs3, SATA) =
**394 MB/s**; NVME1TB (exFAT) = **2.1 GB/s**. The 162.6 GB Q8_0 is copied to
`/media/mal/NVME1TB/qwen38/qwen38-full-q8_0.gguf` for calibration only; the Crucial1TB copy stays
put because it is the `llama-quantize` source and NVME does not have room for the 162.6 GB source
and the ~80 GB output at once.

Working config (`run_imatrix.sh`): `-c 8192 -b 8192 -ub 8192 --chunks 48 -t 12
-ngl 99 --cpu-moe` — 8192 is V4's own sequence length, so this is *closer* to V4 than the
512-token fallback it replaces.

**HELD-OUT LEAKAGE — measured 2026-10-27, and the final gate is optimistic because of it.**

`heldout.txt` and `corpus.txt` share source documents, so the held-out split does not isolate
anything:

| | this run | V4 (measured from `calib-1017-v4/tokens.npy`) |
|---|---:|---:|
| held-out content also in the fit text | **39.9%** of 120-char shingles | 12.2% of 32-token k-grams |
| distinct sequences | 70.6% unique shingles | 200/200 distinct |

**The rule is identical** — "every 4th sequence of each domain" (`v41_calib.py:239-243`,
`prep_calib.py:160-167`). What differs is the condition that made it safe for V4: V4 put **one
long rendered session per sequence** (~8192 tok ≈ 29k chars), giving ~1.2 sequences per document.
This run uses **1843-char sequences sliced out of the same documents**, giving ~4.8 sequences per
document and ~12 non-overlapping slices each. The every-4th-sequence split therefore cuts
*through* documents: one slice of a session lands in fit, the next slice of the *same* session
lands in eval.

Scope: **no effect on the imatrix** (it takes `corpus.txt` only and has no held-out concept; V4's
own agent pool was itself 72.5% duplicated — 1106 texts, 304 distinct) and **no effect on
screening** (the dump has no split either). It affects only the **final gate**: any held-out
perplexity computed from `heldout.txt` is an *optimistic bound*, not a generalisation measure, and
must not be compared to V4's gate as though like-for-like.

**A second, independent mechanism: two pool files are shared across domains.**
`prep_calib.py:46-56`, parsed from the source (beware: two entries wrap across lines, so a
line-by-line regex undercounts): `antirez_general.jsonl` → **`chat` + `web`**, and
`antirez_language.jsonl` → **`reason` + `zh`**; the other 8 files serve one domain each. Each domain
applies the every-4th *per-domain* split, so a language document can be in `reason`'s fit set and,
via `zh`'s independent split, simultaneously in `zh`'s eval set. This one would contaminate even at
V4's sequence length — it is not caused by the slicing change.

**Priority, measured:** only 15 of 274 held-out sequences (5.5%) sit in the shared-pool domains
(`zh` 4, `web` 6, `chat` 5), so cross-domain can account for at most 5.5% — the 39.9% overlap is
dominated by **slicing**. Fix document-level slicing first; global ids second.

**Fix (deferred, not attempted under time pressure):** hold out by **global** document id — one id
assigned to eval puts it in eval for *every* domain that draws on it — then slice and pack. A
per-domain document split would not catch the cross-domain leak. Caveat: `zh` is 16 sequences and
`multi` is 12, so the split needs a per-domain eval minimum *while still* respecting global ids
(promote whole documents into eval until the domain has enough).

**Completion: two independent proofs, and the data one is stronger.** Parse the finished GGUF and
assert `imatrix.chunk_count == 48` **and** `imatrix.chunk_size == 8192` (metadata keys, readable
from the header alone). `save_imatrix()` is passed the actual `m_last_chunk`, so this number tracks
chunks really processed — a run that died at 36 leaves `chunk_count = 36` on disk. Verified on the
`at_24` checkpoint: `chunk_count = 24`, `chunk_size = 8192`, **1852 tensors**, 144/144 expert
entries, 926 `in_sum2` + 926 `counts` pairs. Those are known-good numbers for the final file, not
expectations. The log marker below is the second proof; if the two ever disagree, believe the file.

**A filename- or size-based check is not proof.** `--save-frequency` writes a full
580 MB checkpoint every N chunks, so `imatrix-v4.gguf` exists from chunk 12 onward and is
byte-full-size before the run is done. The only reliable signal is
`grep -q "llama_perf_context_print:.*load time"` — `imatrix.cpp:1184-1188` runs
`save_imatrix()` then `llama_perf_context_print()`, and only after `compute_imatrix()` returns
true. "File exists and is 580 MB" is a *mid-run* state.

### 3.2a Running the engine — four things that each cost an hour (measured 2026-10-27)

The dump is produced by `build-xolotl/strata`. Getting a working invocation took several failed
runs; each of these fails *late* or *misleadingly*:

```
STRATA_DUMP_MOE_ALL=/media/mal/NVME1TB/qwen38/dumps ./build-xolotl/strata \
    --native  /media/mal/NVME1TB/qwen38/qwen38-full-q8_0.gguf \
    --pack    /home/mal/qwen38-pack \
    --ple-gguf /media/mal/NVME1TB/qwen38/qwen38-full-q8_0.gguf --ple-io direct \
    --mmap-experts --spec 2 \
    --prefill 2048 --max-context 49184 --max-new 1 \
    --expert-cache 1536 --expert-profile <profile.bin> \
    --tokens-file <ids-49153.txt>
```

1. **There is no `generate` subcommand.** `strata generate …` prints `unknown argument: generate`
   and `return 2` (`generate.cpp:2247`). The binary *is* generate — `strata --help` even prints
   "strata generate …" as its usage line, which invites the mistake.
2. **`--spec 2` is mandatory for a native pack** (`generate.cpp:3028-3035`, `return 2` otherwise),
   **and it then requires a device residency table**, which is only allocated when
   `hit_fn != nullptr && !profile.empty() && !no_pool` (`generate.cpp:6427`). `hit_fn` needs
   `--expert-cache > 0` (`:5954`), and the profile must have a **non-empty** ranked list —
   an empty ranking leaves the vector empty, so the check fails and you get the *same*
   "`--spec` needs the device residency table" error. This failure happens **after the whole
   prompt has been prefilled**, so it looks like an engine bug rather than a missing input.
   Profile format (`expert_cache.cpp:72-120`): `"STRP"` + `uint32[5]` =
   `{version=1, n_layers=48, n_expert=512, want_slots, n_ranked}`, then `n_ranked` × `2×uint16`
   (layer, expert).
3. **The weight arena is allocated first**, before KV and expert caches (`cudaMalloc(1485606400)`).
   So **only one engine may run at a time** — a second one fails on that first allocation with a
   message naming whatever was free at that instant, which reads exactly like a configuration
   limit. Two incidents tonight were both contention artefacts misread as VRAM ceilings.
4. **Chunk size is nearly free — streaming dominates.** Measured 117.6 s/chunk at 2048 tokens and
   116.0 s/chunk at 1024, because every chunk streams ~all 512 experts regardless of token count
   (host read time was 256 s of a 353 s three-chunk run). So `--prefill 2048` halves the wall clock
   versus 1024 for the same token count, and `STRATA_PREFILL_CPU_SHARE=0` is unnecessary above 1024
   tokens (share is already 0.00 there); it only matters for short preflights.

**Verified dump shape** (24 chunks × 2048 tokens): header `(2048, 2560, 10)`, exactly
`24 + 2048*2560*4 + 2048*10*4 = 21,053,464` B per file, 48 layers × 24 chunks = **1152 files**,
`c0000..c0023` per layer, 24.25 GB total. **T is `--prefill`, so a different chunk size changes
both the header and the byte size** — the gate must be told which was used rather than assuming.

**A resume-only failure mode, worth knowing because it looks like a fresh one.** The dump's
space gate originally ran *before* the "already complete ⇒ skip" check, so a **second** run failed
with "only 16G free (need 24.2G)" — the 24.2 GB of dumps was itself the reason space was low. The
first run created the condition that broke the second. Order must be: count files → 1152 ⇒ reuse;
≠ 0 and ≠ 1152 ⇒ refuse to mix; 0 ⇒ space gate, then dump. The same shape applies to any
"is there room for X" guard placed above "do we still need X".

**The second resume-only failure: a cleanup line destroyed finished work.** Step 4's smoke test
began with `rm -f "$NVME/stats/L00.json"` — written so a stale single-expert file could not make
the layer loop skip layer 0. But the smoke writes to **`smoke-L00-e0.json`**, a distinct name, so
the `rm` protected nothing and deleted a **completed 2-hour layer** on every restart. Found only
on a restart; the first run created the condition that broke the second.

Fixed by **deleting the line**, not guarding it — a guard would be code that exists only to
protect against itself. Two standing notes: (a) `smoke-L00-e0.json` and `L00.json` must stay
distinct filenames; if they were ever "tidied" into one, the smoke's single expert would land in
the real layer file and step 5 would skip layer 0 as done, putting a one-expert layer into the
allocator with no error anywhere; (b) a *resume* rule needs a companion check that nothing VALID
was removed between runs — checking only that nothing partial remains is the mirror image and
neither implies the other.

**A number without its denominator is a bug.** The single-expert smoke test printed
`unmatched_route_rows=1 (0.0020% of 49152)` — a correct count divided by the whole layer's token
count, when only one expert's ~1,000 rows were examined. The layer-wide figure is 341 / 491,520
id slots (0.069%), and 1 for expert 0 is consistent with it. When a run is scoped
(`--limit-expert`), label it and omit the percentage.

**Watching it: an alarm must know what BENIGN looks like in every phase.** Three false positives
were produced in one night by checks that only understood the steady state — a pipeline restart
(step rewound, workers 16→0), `pgrep -f` matching its own command line, and the **tail of the
sliding queue** (33 layers over 16 slots means the last wave runs *one* worker for ~1.6 h, so the
count legitimately falls 16→1). The working rule: gate worker-health checks on `status=STEP5`,
and treat a low worker count as an anomaly **only when layers have also stopped landing**
(`find stats -name 'L*.json' -newermt '-30 minutes'` is empty). Step 5 is a sliding queue, but
with equal-duration layers it degenerates exactly into waves — 16 finish together, 16 start
together — so the wall clock is `ceil(layers/16)` waves, not `total/throughput`. The final lone
layer runs uncontended at the single-threaded 98 min, ~20 min faster than a contended wave.

**A null result needs a positive control.** Every bug found on 2026-10-27 that produced a *wrong*
number was caught by a check asking "is this plausible". None of those checks would catch the last
one: an unscaled `--budget-gib` on a layer subset makes the knapsack return **`+0.00%`**, which is
entirely plausible — it is a *right* number about the wrong question. What caught it was comparing
two runs that **should have differed** and finding them identical. So: before trusting a null
result, run a case where the answer is **known to be non-null**. If the control also returns null,
the tool is broken and the real number is meaningless. A test that can only pass is not a test.

**THE RULE THAT FOUND MORE BUGS THAN ANY OTHER: if a candidate answer beats a known bound, the
candidate is wrong.** Both hard bugs of the late session were *impossible numbers*, and in each
case the impossibility was the signal that nearly got missed for looking plausible:

| candidate | claimed | why impossible | truth |
|---|---|---|---|
| emitted error vs intent | 0.651x (collapse *improved* quality) | emitted is a **strict restriction** of intent *and* spends fewer bytes — both push error up only | 2.710x |
| solver objective | 0.8485 (beat the optimum) | uniform is a feasible point, so the optimum cannot be worse | 0.8805 |

Both bounds are **provable inequalities between two evaluations of the same objective** — no
threshold, no tolerance, no judgement call:

- **Emitter:** `emitted_err >= per_expert_OPTIMUM_err` always — every per-layer assignment *is* a
  valid per-expert assignment, so the per-expert optimum can only be better. **It is NOT
  `>= intent_err`**: intent is the allocator's *answer*, and substituting an answer for an optimum
  is a different (and unprovable) claim. At this budget they happen to coincide —
  dual bound 0.4993, feasible 0.4994, allocator reports 0.4993, i.e. the allocator is **optimal
  within 0.01%** — so `emitted >= intent` holds here with 0.01% margin, as an empirical coincidence
  rather than a theorem.
- **Solver:** `per_layer_err <= uniform_err` always, because assigning the same combo to every
  layer reproduces uniform exactly, so uniform is inside the feasible set.

A violation of the **optimum** bound is a measurement bug by definition; plausibility must not
survive it. This is strictly stronger than a threshold check, which cannot distinguish "spent
similar" from "reproduced the allocation" — a fixed byte bound passes on the −7% under-spend and any
threshold tight enough to fire also fires on a legitimate saving.

**A bound that is stated against the wrong reference is itself the bug.** A claimed "per-expert
optimum" of 0.4605 was reported during this analysis; it sits **below** the dual lower bound of
0.4993, which is impossible — so it was flagged as a measurement bug. That is why the allocator must
**not** be "replaced with an exact solver": once the comparison is fair it is optimal to **0.02%**.

**The cause was an under-specified ITEM SET, not a wrong byte count — and that is a new failure
mode.** The optimizer that produced 0.4605 dropped the 168 **unseen** experts entirely: their error
(right — the allocator zeroes it) *and* their bytes (wrong — the allocator still pays for them,
flooring them at their tier). Their 0.2043 GiB is **1.49%** of the budget, and handing 1.49% more
bytes to every seen expert buys **−7.79%** error. Reproduced exactly:

| item set | proxy @ 13.750 GiB |
|---|---:|
| unseen dropped (item **and** bytes) | **0.4605** |
| correct (unseen included, err = 0) | 0.4994 |
| `v41_alloc` reports | 0.4993 |

**So "state your bytes" is not sufficient — the invariant is three clauses**, and each of the three
bad numbers produced in this hour violated exactly one:

1. **same budget** — bytes stated and equal (violated by the 19.75 / 13.855 / 14.011 / 9.33 cases);
2. **same item set** — every expert contributes, *including those with zero error*
   (violated by 0.4605: the byte count was correct to the byte, 13.749981 GiB, and still wrong);
3. **same objective** — `w * err / energy`, edge-weighted, unseen→0, applied identically to both
   sides (violated by the 0.651x scope error).

**Clause 3 is the one that keeps biting, and the reason is structural.** Across eleven instances
tonight it fired three times (the 0.651x scope error, the Pareto-sweep explanation, and two of the
peer's own tables quoting *ratio-of-sums* and *mean-of-ratios* under the same name "the per-layer
ratio"). The three clauses differ in how loudly they fail:

| clause | failure mode | loudness |
|---|---|---|
| budget | numbers stop adding up | **loud** |
| item set | visible if you enumerate the items | **visible** |
| objective / aggregation | both numbers are correct, and both about the same thing *in different senses* | **silent** |

That is why the third needs an explicit naming habit rather than care: no arithmetic check can catch
it, because there is nothing arithmetically wrong. **Name the aggregation or the number is
ambiguous**: `Σq2_0/Σiq2_xxs` (a layer's *total* error) is not `mean(eᵢ)` (each expert weighted
equally) — they differ by ~50% in spread on identical data, and the choice manufactured a
non-existent L05 outlier that a pre-registered test would then have been run against.

**A wrong number that becomes a wrong test design** is the worst outcome in this file: a mismatch
would have been read as "proxy and perplexity disagree" — the exact false conclusion the discipline
exists to prevent.

A related trap, mine: a Pareto sweep over the *correct* frontier placed 0.4605 at "13.954 GiB,
over budget" — but 0.4605 is simply unreachable on that frontier, so the sweep found the nearest
reachable point and the difference was attributed to bytes. **Right answer, wrong reason** — and the
wrong reason would have sent someone hunting a budget bug that did not exist.

**That is a distinct failure class worth naming separately: the error was in the *explanation*, not
the measurement.** Every other mistake in this session produced a bad *number*; this one produced a
bad *causal claim about a correct number*, with a plausible fit attached to it. It matters more than
a wrong number, because **a number gets re-measured while an explanation gets written down.** The
discipline is the same — test the mechanism, never infer it from a plausible fit — but the blast
radius is different, so a diagnosis deserves the same falsification a number gets.

**The control itself had four bugs in one hour** — build one and assume nothing:

| failure | how it looked | found by |
|---|---|---|
| control **skipped** | "no uniform config fits" | ordinary construction attempt |
| control **always failed** | `+0.00%` at exactly the cheapest uniform — arithmetically *correct* there | second attempt |
| control failed for the **wrong reason** | inherited a non-binding main budget, so it stopped binding | capping it |
| **passed on a broken solver** | `abs(adv) < 1.0` accepted **−7.73%** | deliberately degenerating the solver |

The fourth is the one that matters: a negative advantage is **impossible** for a correct solver,
because uniform is a feasible point of the per-layer knapsack (choosing the same combo for every
layer reproduces it exactly), so `per_layer_error <= uniform_error` always. A negative result is a
*proof* of a bug — and `abs()` turned the proof into a pass. Assert the **invariant**
(`adv >= 0`) separately from the **threshold** (`adv >= 1%`): the first is a hard failure with a
mathematical reason, the second is merely "this control is too weak to discriminate". They must not
print the same way. (The invariant holds only while the solver's feasible set contains the uniform
assignment; the Lagrangian's **+0.40% overshoot** is what keeps it sound — a future fix that clamps
*downward* would void it.)

**The method that found it, which is the transferable part: break the thing on purpose and check
the alarm sounds.** Degenerating the solver took minutes and exposed a bug no amount of ordinary
use would surface, because a broken solver looks exactly like a healthy one from outside. Do this
for every check: introduce the fault it exists to catch, confirm it fires, remove the fault.

Two further corollaries from the same night:

- **Agreement between two implementations is a positive control too.** The peer's knapsack and mine
  disagreed on the *picks* while agreeing on the percentage — that disagreement is what exposed a
  missing `energy` normalisation and a missing unseen-zero term. Comparing only the headline number
  would have hidden both. **Compare the ranking, not just the magnitude.**
- **Contention during a measured run costs more than its share.** 16 workers on 8 physical cores is
  already oversubscribed, so a 17th CPU-bound process does not take 1/16 of throughput — it takes a
  slice of an already-shared core from everyone. Answer questions from the source when you can, and
  do not run benchmarks beside the thing being measured.
- **A constant describing the MODEL must never be inferred from the data subset.** Three instances:
  `budget_bytes` sized for 48 layers applied to 15 (over-upgrades, catastrophic on small N);
  `EDGE_HI` derived from the loaded subset instead of `n_layers` (silently mis-weights the last
  three layers); `--n-layers` defaulting to `max(present)+1`. Audit for this by asking of every
  constant: *does this describe the model or the data?*

### 3.2b The pack (`iq_pack.py`) does not touch the experts

`python3 tools/iq_pack.py --gguf <q8 gguf> --out <dir> --compat-bf16`. The help text says it
"dequantize[s] small non-native projections to BF16 … **leaves experts and the PLE table
unchanged**", so `dense.bin` is ~5 GB and the 128 GB of expert weights are never copied — they are
read from the GGUF by offset via `native_experts.txt`.

Three independent reasons the BF16 rounding is inert — and note the *second* one is not free,
because `X` feeds two places:

1. **The weights are not involved.** `screen.py` reads expert weights straight from the FP8
   checkpoint via its own `FP8Model`; `--compat-bf16` only affects the dense path that produces
   the *activations* `X`.
2. **The one path that IS first-order was measured and does not matter.** The error comparison
   `(Yq - Y)` is common-mode in `X` and largely cancels — but `im_gu = (|Xe[fit]| * wgt).mean(0)`
   is built *from* `X` and then fed to the quantiser, so a 0.39% activation perturbation becomes a
   0.39% per-element perturbation of the **importance weights**, which is not common-mode.
   Measured anyway, 8 draws per tier through the real ggml quantiser: worst-case change in the
   measured error is **0.0045%** (`iq3_s` ratio range 0.999960–1.000045), four orders of
   magnitude below the smallest tier gap (IQ4_NL 0.076 vs IQ3_S 0.143 ≈ 47%). The IQ lattice
   argmin does not move.
   *(Two false starts worth not repeating: a **uniform** scale of the imatrix changes nothing at
   all — it cancels in a relative-weighted search — and 0.01% noise is an order of magnitude too
   small to test with.)*
3. **It never reaches the shipped file.** `run_quant.sh` quantises from `SRC`, the Q8_0 GGUF
   (`SRC=…/qwen38-full-q8_0.gguf`, passed at line 73), **not** the pack. The pack exists only for
   the engine dump. So rounded dense cannot contaminate the XQ output at all.

### 3.3 Allocate

```
python3 tools/v41/v41_alloc.py --arch qwen4exp --per-proj \
    --work <stats dir> --edge-weight 2.0 --min-eval 8 --unseen-floor iq2_xxs \
    --budget-gib <target> --out alloc.json

python3 tools/qwen38/emit_tensor_types.py alloc.json --per-proj -o tensor_types.txt
```

The allocator needs `stats/L*.json` with per-expert, per-tier held-out error.

**`--unseen-floor` must be `iq2_xxs` here, NOT `q2_0` — MEASURED 2026-10-27.** The guard
(`v41_alloc.py:66-70`) skips every tier whose bpw is below the floor, reasoning that "cheaper
bpw = riskier tier". On DS4's ladder (`ptq1/q2_0/q3_k`) that held. **qwen4exp's ladder is
non-monotonic**: summed over the first 16 layers on real weights,

| tier | bpw | Σ `egu_` error |
|---|---:|---:|
| IQ2_XXS | 2.0625 | **4,644.8** |
| Q2_0 | 2.2500 | **24,389.5** |

Q2_0 is **5.25x worse while costing more bits** — it is a plain scalar quantiser with no codebook.
So `--unseen-floor q2_0` *excludes* IQ2_XXS for unseen experts and forces them onto Q2_0, which is
both larger and 5x worse. That is backwards from the guard's intent. With `--unseen-floor iq2_xxs`
the tier is no longer excluded (2.0625 is not < 2.0625) and the solver picks it for unseen experts,
since their error is 0.0 for every allowed tier and the choice therefore falls to cheapest bytes.

Measured cost of the wrong flag: 223 lowdata experts in 16 layers (~669 across 48) get ~51 MB
larger and 5.3x worse gate/up. The bytes are trivial; the quality is not.

**`--per-proj` is also mandatory, and silently so.** Without it the allocator reads `e_{t}` for the
combined part — and `screen.py` only writes `e_{t}` for tiers that exist on **both** ladders
(`q2_0`, `iq4_nl`). The other five keys are absent, and `v41_alloc.py:71` reads a missing tier as
**`0.0` error**, i.e. perfect and free, so the solver would select an unmeasured tier everywhere.
With `--per-proj` it correctly reads `egu_{t}` (all 7) and `edn_{t}` (the 2 the down ladder
supports).

**Sizing the budget against the 77 GiB gate — measured from the source GGUF's tensor table
(1224 tensors), applying quantize's real `tensor_allows_quantization` rules:**

```
routed experts               120,795,955,200 params      -> the --budget-gib bytes
dense, quantizable             4,883,742,720 params      -> Q8_0    =  4.833 GiB
dense, NOT quantizable            63,955,840 params      -> F32     =  0.238 GiB   (304 tensors)
per_layer_token_embd          51,200,245,760 params      -> IQ4_NL  = 26.822 GiB
                                                           fixed subtotal = 31.893 GiB
```

The 304 unquantizable tensors are **F32 in the source** — verified: the source GGUF has 388 F32
tensors, of which 304 fail `tensor_allows_quantization` (`ggml_n_dims < 2`, `_norm.weight`,
`ffn_gate_inp.weight`, non-`weight` names). A blanket "all dense at 8.5 bpw" understates the file
by **0.173 GiB**, which is exactly how an earlier draft of this table got 31.720 GiB.

`--budget-gib` is **expert bytes only**, so the whole file is `budget + 31.893 GiB`:

| budget | whole file | vs 77 GiB gate |
|---:|---:|---|
| 47 | 78.893 GiB | **FAIL** |
| 46 | 77.893 GiB | **FAIL** |
| 45 | 76.893 GiB | pass, but only 0.107 GiB margin |
| **44** | **75.893 GiB** | **pass, 1.107 GiB margin** |

The exact ceiling is 45.11 GiB. Use **44** — 45's margin is thinner than GGUF alignment slack
should be asked to absorb. Effective expert bpw is 3.13 at 44 GiB (vs 3.34 at 47) — still below
Q3_K_S, so the file is still smaller than UD-Q4_K_XL's 77 GiB. Sanity check on the counts:
176.94e9 params in a 162,606,395,072-byte file = 7.34 bpw, exactly what Q8_0 experts+dense with
IQ4_NL PLE should average.

**The all-Q2_0 baseline is 63.534 GiB** — predicted by simulating the emitted rules against all
1224 real tensor names (144 expert rules → Q2_0 31.641 GiB, PLE IQ4_NL 26.822, dense Q8_0 4.833,
F32 0.238), which reproduces the 31.893 GiB subtotal a third way. Two consequences:

- `--budget-gib 44` is **12.4 GiB above** that baseline, so the allocator is upgrading a large
  fraction of experts off Q2_0, not trimming. That is the intended use of the gate headroom.
- If the allocator ever reports `used` well under the budget, that is a red flag — it should fill
  it. Under-filling means the ladder topped out or the edge-weight/ternary caps bit.

**`--budget-gib` is sized for the layer count, so it must be rescaled for any subset.** MEASURED:
running 44 GiB against only 15 of 48 layers used just 19.53 GiB and pushed ~95% of experts to
IQ4_NL (4.42–4.49 bpw) with **zero** on any mid tier — it had 24 GiB of unspent headroom and
correctly upgraded everything to the best available tier. The same 15 layers at the properly
scaled 13.75 GiB give a real XQ shape (iq2_xxs ≈200/layer, q2_0 ≈90, iq2_s ≈100, iq3_xxs ≈110,
iq3_s ≈50, iq4_nl ≈450, bpw 2.9–3.3). Assert `len(stats) == 48` when the budget is 44, so a
partial run fails instead of silently over-upgrading.

**The `--unseen-floor iq2_xxs` fix, verified on real data.** Same 15 layers, same 13.75 GiB, only
the floor changed:

| floor | unseen experts → gu | bytes | proxy error |
|---|---|---:|---:|
| `q2_0` (old) | 168 × `q2_0` | 13.75 GiB | 0.5017 |
| `iq2_xxs` (new) | 168 × `iq2_xxs` | 13.75 GiB | **0.4993** |

Down stayed `q2_0` in both. So the change moves 168 experts to the *smaller* tier and **lowers**
the proxy error 0.48% on an identical byte budget — the dominated choice, confirmed by measurement
rather than by the bpw table.

**Two diagnostics in that output are not data.** The `ratio(GPTQ/RTN)` column reads `0.00/1.00`
for six of seven tiers because the ggml path has no GPTQ variant — uninformative on this arch, not
a fault. And the "−96.6% vs all-Q2_0" proxy improvement is flattering because all-Q2_0 is a
terrible baseline on a non-monotonic ladder; do not quote it as a quality claim.

### 3.4 Quantise

```
llama-quantize --imatrix imatrix.gguf \
    --tensor-type-file tensor_types.txt \
    --allow-requantize \
    model-Q8_0.gguf Qwen3.8-Flash-Next-XQ.gguf Q8_0
```

`--tensor-type-file` patterns are compiled as C++ `std::regex`, so the emitter anchors them
(`^blk\.20\.ffn_(gate|up)_exps\.weight$`) and a layer-2 rule cannot leak onto layer-20. **Rules are
first-match-wins** (`llama-quant.cpp:716` breaks on the first `regex_search` hit), which is why the
emitter puts the expert rules first, then the PLE rule, then the dense catch-all.

**Verified against real C++ `std::regex`, not Python `re`** — by
`tools/qwen38/check_tensor_types_regex.cpp`, which compiles every pattern exactly as
`llama-quant.cpp:196` does:

```bash
g++ -O2 -std=c++17 -o /tmp/check_tt tools/qwen38/check_tensor_types_regex.cpp
/tmp/check_tt <emitted tensor_types.txt>     # exit 1 = a pattern failed, or an expert is untyped
```

Run it on the **actual emitted file** after any `emit_tensor_types.py` change. Python `re` and
`std::regex` ECMAScript diverge in general (`(?P<name>...)`, variable-length lookbehind), so a
pattern the emitter is happy with can make llama-quantize throw at startup — hours into an
unattended job.

**The harness identifies the dense catch-all by RULE INDEX, not by type.** Its first version
flagged an expert only when it matched *no* rule, which is a false negative: an expert that falls
into the catch-all *did* match a rule. Demonstrated with a layers-0-7-only rule set plus a broken
catch-all set to IQ4_NL — `blk.47.ffn_gate_exps.weight` resolved to `IQ4_NL` via the catch-all and
the check passed it. Now the catch-all index is resolved by probing a tensor only the catch-all can
serve (`blk.0.attn_q.weight`), and an expert is flagged if `idx < 0` **or** `idx == catchall`.

**Probes are generated for all 48 layers x 3 projections, not sampled.** A blk.0/blk.47 sample
passes while `blk.23` has no rule at all — exactly what a partial rule set (`--layers 0-7`)
produces. 144 synthetic names cost nothing and are exhaustive over the rule space; whether the GGUF
*contains* those tensors is a separate check (the peer's gate b2 walks the real tensor list).
The two are complementary: this harness proves the **dialect** compiles (the one thing Python `re`
cannot tell you) and that every layer/part rule exists; b2 proves the GGUF's actual 144 expert
tensors are each individually ruled, plus the size ceiling/floor.

Verified failure modes, all exit 1 and named:

| case | result |
|---|---|
| healthy 98-pattern file | 144/144 expert probes ruled, exit 0 |
| `blk.23` rules deleted | 3 flagged `<-- NO RULE` |
| layers 0-7 only + broken catch-all `IQ4_NL` | 136 flagged `CAUGHT BY THE DENSE CATCH-ALL` |
| `^blk\.(?P<layer>\d+)\.attn_q\.weight$` | `COMPILE FAILED (Invalid '(?...)' zero-width assertion)` |

**The catch-all must never match an `*exps` tensor — under *any* `--dense-type`.** The anchored
form `^blk\.(?!.*ffn_(gate|up|down)_exps\.weight).*$` does not match experts *at all*, so they fall
through to llama.cpp's own `default_type` (`llama-quant.cpp:708`, set by the final ftype argument,
independent of `dense_type`). The original `^blk\..*\.(?!ffn_…)…$` had a dead lookahead and **did**
match experts: with `dense_type = Q8_0` that was invisible, but with `--dense-type IQ4_NL` it would
send every unruled expert to 4.5 bpw — ~68 GB instead of ~32 GB on 120.8e9 expert params, blowing
the 77 GiB gate in a way that looks like a ladder bug rather than a regex bug. Demonstrated in C++
in isolation.

**`--allow-requantize` is mandatory here** and was missing until 2026-10-27. The source is Q8_0 for
every tensor, and `llama-quant.cpp:1253` throws `requantizing from type Q8_0 is disabled` on the
first expert tensor otherwise. The Q8_0 step's own error is 0.552% against FP8 whose own
quantisation error is ~2.55%, so requantising from it is inside the precision this pipeline already
assumes.

**The PLE table must be pinned.** `per_layer_token_embd.weight` is already IQ4_NL (28.8 GB) and
does not start with `blk.`, so it would otherwise fall to the Q8_0 default — and with
`--allow-requantize` that is a legal re-expansion to ~51 GB that quietly blows the size gate. The
emitter therefore writes `^per_layer_token_embd\.weight$=IQ4_NL` before the catch-all.

**Threads: `nthreads` is the 5th POSITIONAL argument, and omitting it is correct.**
`quantize.cpp:126` — `model-f32.gguf [model-quant.gguf] type [nthreads]`. There is no `-t`; passing
`-t 8` does not set the thread count. `run_quant.sh` passes no `nthreads`, which is right because
`llama-quant.cpp:897-901` does `if (nthread <= 0) nthread = std::thread::hardware_concurrency();`
— 16 here, not 1. It really does thread per expert slab: `nthread_use = min(nthread, nchunk)` at
`:1300`, and for a 2560-wide gate row `chunk_size` is 17,920 elements, so `nrows_cur=128` gives
`nchunk=19` → **16 threads**. Expect ~1.4 h, plan 2 h. **If step 9 overruns, check the
`using N threads` line in the log first** — if `hardware_concurrency()` ever resolves to 1, that
step becomes ~11 h.

*(Do not try to measure this scaling while imatrix is running: a 1-thread vs 8-thread test on a
local model gave 4163 ms then 1062 ms on the same configuration. That measured disk contention on
shared volumes, not threading. The code path above is the evidence.)*

---

## 3.5 Block sizes decide the ladder (measured, not assumed)

A ggml type can only be written for a tensor whose row length is a multiple of the type's block
size. qwen4exp's `n_embd` is 2560 and `n_ff` is 640. Measured through ggml itself
(`gq_row_bytes`):

| tier | block | gate/up (2560) | down (640) |
|---|---:|---|---|
| IQ2_XXS / IQ2_XS / IQ2_S | 256 | yes | **no** |
| IQ3_XXS / IQ3_S | 256 | yes | **no** |
| Q3_K / Q4_K / Q5_K / Q6_K | 256 | yes | **no** |
| Q2_0 | 64 | yes | yes (180 B/row) |
| IQ4_NL | 32 | yes | yes (360 B/row) |
| Q8_0 | 32 | yes | yes (680 B/row) |

So **down has exactly two sub-8-bit choices: Q2_0 and IQ4_NL.** That is not a heuristic choice by
the third-party quants — it is the format. `v41_alloc.py --arch qwen4exp` therefore carries a
separate `dn_tiers` ladder, and `screen.py` drops unusable tiers per projection and prints what it
dropped rather than reporting an error for a file that cannot exist.

Gate/up keep the full seven-step ladder.

## 3.6 Where the per-expert error comes from

`screen.py` measures, per expert, the output error of each candidate tier on the tokens that
actually route to it, and writes `stats/L<NN>.json` in the schema `v41_alloc.py` already reads.
Quantisation goes through **ggml itself** (`ggml_quantize_chunk`, imatrix-aware) via
`ggml_qshim.cpp`; a numpy re-implementation of IQ2/IQ3 would measure a different format.

Inputs are the engine's own MoE dumps (`X` = pre-MoE hidden state, `ids` = routed experts). The
router weights are recomputed from `X` and the router tensor, so nothing extra has to be plumbed.

**Weights are keyed by expert id, not by slot (fixed 2026-10-27).** The engine's
`route_topk` (`src/kernels/cuda/native_router.cu:185-200`) selects the max at every rank, so its
`ids` are **sorted by descending weight**; `np.argpartition`'s within-k order is arbitrary.
Measured slot agreement between the two: **44.8%**. Pairing `w[rows, kcol]` against a dumped id
therefore attached the wrong expert's weight on most rows. `screen.py` now scatters into a full
`(T, n_router)` matrix and indexes `w_full[rows, e]`, which is exact under either ordering — verified
by writing two dumps with identical per-row expert *sets* in different slot permutations and
getting bit-identical tier errors.

The impact was **small but real**: on real L0 router + expert 0, the buggy pairing changed each
tier's error by at most **2.1%** and did **not** reorder any tier. The reason is that the measured
quantity is a ratio whose numerator and denominator carry the *same* weights
(`Σ w²‖f_q−f‖² / Σ w²‖f‖²`), so a wrong weight re-weights a ratio rather than scaling one side.
Still worth fixing — it is a measurement error, and the fix costs ~100 MB.

**A non-finite error drops the expert, not the layer.** A NaN/inf in one expert's error would reach
`v41_alloc.py` as an ordinary float, and `min()` over NaN picks unpredictably — one bad expert could
silently steer a whole layer's tiers. `screen.py` now checks every error and `y_energy` after the
tier loop and **drops** a bad expert, recording `summary.n_dropped_experts` / `dropped_experts` and
warning on stderr. Dropping (rather than raising) is deliberate: an absent expert is *unseen* to
`v41_alloc.py:61` and gets floored at `--unseen-floor q2_0`, the conservative outcome — whereas
aborting would cost the other 511 experts in that layer, and in an unattended chain would stall the
whole run over one poisoned row. Verified: clean data → 0 dropped; one NaN in a routed row → that
expert dropped, file still written, exit 0.

**The guard tests the numeric ABC, not `float`.** `isinstance(np.float32(nan), float)` is **False**
in numpy, so a `isinstance(v, float)` filter would let a numpy-typed NaN through and disable the
guard silently, with no test failing. The predicate is therefore
`isinstance(v, (int, float, np.integer, np.floating)) and not np.isfinite(v)`, and the `y_energy`
test is `not np.isfinite(y) or y <= 0` (because `nan <= 0` is False). Tested across py-float,
np.float16/32/64, int, np.int64 and bool — all ten cases correct.

**The scatter makes a second failure mode visible instead of silent.** `w_full` is nonzero only at
screen.py's own recomputed top-k, so if the dump routes a row to an expert that screen.py's
recomputation missed, `wgt` is exactly `0` and that row drops out of both numerator and
denominator. That is a *set* mismatch, which the permuted-dump test cannot see by construction.
`summary.unmatched_route_rows` now counts it and warns on stderr. Verified both ways: random ids →
34 unmatched, engine-consistent ids → **0**. On a real dump this doubles as the end-to-end test
that the checkpoint router (used by screen.py) and the GGUF router (used by the engine) agree as
sets — the one comparison the byte-offset reader could not make. If L00 reports `0`, they agree on
all 49,152 tokens.

**The two routers are in fact bit-identical — measured, not assumed.** Reading
`blk.0.ffn_gate_inp.weight` straight out of the GGUF (type 0/F32, `alignment=32`,
`data_start = align(header_end) = 11,023,552`, offset 2,749,622,656) and comparing against the
checkpoint's `mlp.gate.weight`: **`np.array_equal` → True across all 1,310,720 values,
`max|diff| = 0`**, and identical top-10 ids slot-for-slot on 2,048 tokens. The tensor is provably
the same — do not re-litigate it.

**But `unmatched_route_rows` is NOT 0 on a real dump — it is ~341 for L00 (0.69% of rows).**
Measured on the actual 24 chunks: 341 of 491,520 routed id slots are absent from screen.py's own
recomputed top-k. The mechanism, measured rather than guessed:

- top-1 agrees on **2048/2048** rows — the max logit is identical everywhere;
- on the mismatching rows the engine's set is a **subset** of the recomputed one, differing at
  exactly one slot;
- the rank-K vs rank-K+1 margin on mismatching rows has median **1.08e-4**, versus **5.86e-3** on
  matching rows — 54x smaller.

So the disagreement is entirely at the rank-10/11 boundary where two logits sit within ~0.01% of
each other, and the **engine's accumulation order/precision flips those ties**. The router weights
are identical; the *arithmetic* differs. Consequence: `unmatched_route_rows` measures
engine-vs-checkpoint **accumulation**, not router identity — and a margin test at 1e-6 is not
sufficient to guarantee a stable set, because these flips happen at ~1e-4.

Impact is benign: those rows are true ties, so their routing weights are nearly equal, and
dropping them biases the measurement by 0.069% of slots. Expect the same order on every layer.
A gate that flags above 0.5% should be raised to ~2%.

**The ggml layout trap that nearly faked a mismatch.** The GGUF declares this tensor
`(2560, 512)`, but ggml stores `ne[0]` **contiguously**, so the correct numpy read is
`flat.reshape(512, 2560)` — the checkpoint's own shape. Reshaping to the declared order and
transposing gives cosine ≈ 0 and `max|diff|` 2.38, i.e. it *looks* like two unrelated tensors.
The tells that it was a read error and not a real difference: the Frobenius norms matched to six
decimals (28.693433) and the sorted values matched exactly — a permutation, not different data.

**Held-out, V4 parity (fixed 2026-10-27).** The imatrix the quantiser is fitted with is built
from the **fit** tokens and the output error is measured on the **held-out** tokens, exactly as
`tools/v41/v41_calib.py:418-428` does. Before the fix both came from the eval rows, so a gate
tuned on the same rows it was scored on — the split was cosmetic. Two related details now match
V4: `--min-fit` (default 32, separate from `--min-eval`) gates the data-built imatrix, and below
it the imatrix falls back to an **identity Hessian** (`lowdata`) rather than to the eval rows;
`n_eval` is recorded **pre**-expansion so the allocator's `min_eval` confidence floor can fire on
sparse experts.

**No custom kernels are needed anywhere in this pipeline.** `llama-imatrix` is arch-agnostic and
takes `-ngl`; `ggml_quantize_chunk` is already exported. The work is data plumbing, not CUDA.

Validated on a synthetic checkpoint: the measured ladder is monotone and physically correct
(IQ4_NL < IQ3_S < IQ3_XXS < IQ2_S < IQ2_XS < IQ2_XXS < Q2_0 — Q2_0 losing to IQ2_XXS at *more*
bits is expected, it is a plain scalar quantiser with no codebook).

## 4. The granularity limit — CORRECTED 2026-10-28: there isn't one

> **This section previously claimed per-expert tiers "cannot be expressed in a plain GGUF". That is
> WRONG.** It was inferred from reading `llama-quantize` (which does store one type per tensor)
> rather than from inspecting the artifact V4 actually shipped. **V4's shipped GGUF carries
> per-expert types**, via a split-by-type layout. See §4.0. The collapse described below is a
> limitation of *our emitter*, not of GGUF.

### 4.0 How V4 actually did it (inspected in the real file, 2026-10-28)

`/media/mal/NVME1TB/Models/V41-XQ-1017-v4.gguf` — its own metadata:

```
strata.v41.expert_layout = uniform role: ffn_<role>_exps.weight;
                           mixed:   ffn_<role>_exps.<type>.weight per type (experts in id order)
                                    + I32 ffn_exps_{gu,down}_type[384]
```

and the tensors are physically there (layer 3, verbatim):

```
blk.3.ffn_gate_exps.q3_k.weight     Q3_K     dims=(5120, 2304, 50)     <-  50 experts
blk.3.ffn_gate_exps.mxfp4.weight    MXFP4    dims=(5120, 2304, 19)     <-  19 experts
blk.3.ffn_gate_exps.q2_0.weight     Q2_0     dims=(5120, 2304, 82)     <-  82 experts
blk.3.ffn_gate_exps.ptq1_0.weight   PTQ1_0   dims=(5120, 2304, 233)    <- 233 experts  (sum 384)
blk.3.ffn_exps_gu_type              I32      dims=(384,)   <- per-expert type index
```

**Four tensors per projection, one per type, experts grouped, plus an I32 expert→type array.**
Type ids (`tools/v41/v41_formats.py`): `143=PTQ1_0, 42=Q2_0, 11=Q3_K, 39=MXFP4`.

Across all 40 layers × 384 experts (gu): **PTQ1_0 63.3%, Q2_0 18.2%, Q3_K 11.6%, MXFP4 6.9%** —
**four types in every layer.**

**The killer confirmation:** PTQ1_0 (the *cheapest* tier) is the majority in **all 40 layers**. So
applying *our* majority-vote collapse to *V4's own allocation* would produce **PTQ1_0 everywhere** —
the exact bug found below, on the artifact that worked. V4 avoided it only by writing the mixed
layout. **The collapse rule was never necessary.**

Port target: `tools/v41/v41_gguf.py` (`experts_json`, the `mixed` branch) already implements this
for ds4.

### 4.1 The collapse defect (still real — it is our emitter's, not the format's)

`llama-quantize` stores **one ggml type per tensor**, and a layer's 512 routed experts are a single
3D tensor per projection, so `emit_tensor_types.py` collapses to one type per layer by majority
vote. With the mixed layout (§4.0) that collapse is unnecessary — but as written it is actively
harmful:

**Mechanism, measured:** a budget-constrained allocation is mode-heavy at the cheap end, so **the
mode IS the cheapest tier** — on all 15 layers, mode bpw 2.0625 vs mean bpw 2.41–2.69.

That collapse is lossy, and measurably so — on a synthetic run **128 of 128** layer/part groups
contained more than one tier before collapsing. In other words, per-expert allocation buys nothing
for Qwen in a standard GGUF; the decision degrades to a per-layer vote.

**CONFIRMED ON REAL DATA, AND THE COST IS LARGE — 2026-10-27.** Two independent computations
(L1–L15, real screening stats, floor `iq2_xxs`, ~13.75 GiB budget) agree on the shape:

| allocation | proxy error | bytes |
|---|---:|---:|
| allocator's per-expert intent | 0.507 | 13.75 GiB |
| **what the emitter actually ships** | **1.355** | 12.79 GiB |
| best *global uniform* at same budget (`iq2_xs`/`iq4_nl`) | 1.120 | 13.35 GiB |
| best *per-layer* knapsack (`iq2_s`/`iq4_nl` ×13, `iq2_xxs`/`iq4_nl` ×2) | **0.880** | 13.73 GiB |

The emitted rules were `IQ2_XXS` on 13 of 15 layers and `IQ2_S` on 2, `IQ4_NL` for every down —
i.e. **effectively a single uniform configuration**. So the collapse costs **+167% error for −7%
bytes**, and the shipped file is **21% worse than the best uniform config at the same budget** —
we would spend the whole screen and the whole quantize to produce something a uniform choice
beats. A per-layer knapsack is **+21% better than uniform** and **+35% better than the collapsed
path** (the peer measured +27% / +44% with a different normalisation; same shape).

**The objective must match the allocator's, or the comparison is between two questions.** Measured
on the same 15 layers and budget, three conventions give three different sets of picks:

| objective | tier picks |
|---|---|
| raw per-expert sums, no edge weight | `iq2_xxs/iq4_nl` ×4, `iq2_s/iq4_nl` ×10, `iq3_xxs/iq4_nl` ×1 |
| energy-normalised, no edge | `iq2_xxs/iq4_nl` ×2, `iq2_s/iq4_nl` ×13 |
| **energy-normalised + edge 2.0** (matches `v41_alloc.py:58,75`) | `iq2_xxs/iq4_nl` ×2, `iq2_s/iq4_nl` ×13 |

So a knapsack run on raw sums spreads across four combos while the allocator's own objective keeps
13 of 15 layers on one. The allocator divides by per-layer `y_energy` precisely so every layer
counts the same regardless of output scale; dropping that term lets high-energy layers dominate the
sum. **Any per-layer knapsack presented as "the" number must use `w * err / energy`
(`v41_alloc.py:75`) and state it** — otherwise part of its advantage is "a different objective
scores better on itself". Ranking (the picks) is the decision variable; magnitude is the report.

**Per-layer needs no format work.** The emitter already writes one rule *per layer*
(`^blk\.N\.ffn_(gate|up)_exps\.weight$`), so per-layer allocation is expressible today — what has
to change is the *granularity of the decision* in `v41_alloc.py`, from per-expert-then-collapse to
a per-layer multiple-choice knapsack. That is real code in another lane's file, not a flag.

**THE HEADLINE: the collapse RULE is the bug, not the granularity.** The majority vote does not
merely approximate the allocation — it systematically selects **the cheapest tier on every layer**,
because a budget-constrained allocation pushes most experts to the cheap end, so the *mode* of the
distribution **is** the cheapest tier. Measured on all 15 layers: mode bpw **2.0625**, mean bpw
**2.41–2.69**, on every one. That single fact explains both symptoms — it under-spends the budget by
7%, and (on this non-monotonic ladder) it lands worse than a plain uniform choice.

| collapse rule | bytes | proxy | fits 13.75 GiB? |
|---|---:|---:|---|
| **MODE — what ships today** | 12.79 GiB | **1.3533** | yes, −7% (under-spends) |
| best global uniform (`iq2_xs`/`iq4_nl`) | 13.35 GiB | 1.1192 | yes |
| **per-layer / budget-aware** | 13.73 GiB | **0.8805** | **yes — the optimum at this budget** |
| ~~nearest-mean rule~~ | 14.01 GiB | 0.7916 | **NO — 1.9% over budget, not comparable** |

**Do not quote the nearest-mean row.** It looks like the best answer and it is an infeasible
allocation. It was quoted here once as "within 4% of per-layer", which was wrong for exactly the
reason this file warns about elsewhere: **a comparison that does not hold the binding constraint
fixed is not a comparison.** Three separate "better" numbers appeared during this analysis (a
19.75 GiB "best uniform", a 13.855 GiB hand-DP, and this 14.011 GiB mean-matched) and **all three
were feasible-looking assignments at the wrong byte count.** Any allocator comparison must state its
bytes; a quality number without one is not a result.

At equal bytes the only improvement over uniform is the per-layer assignment — a naive bpw-mean
heuristic does not buy it more cheaply, it reaches similar quality by spending 2% more. So **the
budget-aware rule *is* the per-layer knapsack**, and "one line" understates the change: the honest
description is *replace mode with a budget-aware tier choice*, expected gain **1.3533 → 0.8805, a
35% error reduction**, at essentially the same bytes and marginally better than the best uniform.

### Guarding the emitter: assert quality, not spending

A byte bound is the obvious safeguard and it is **blind to this bug**, because the collapse
*under*-spends:

| proposed assertion | buggy pipeline | verdict |
|---|---|---|
| `emitted_bytes <= budget` | 12.792 ≤ 13.75 | **PASS** — misses it |
| `emitted_bytes >= 0.90 * intent_bytes` | 12.792 ≥ 12.375 | **PASS** — misses it (under-spend is −7.0%, threshold is −10%) |
| `emitted_err <= 1.05 * intent_err` | 1.3533 vs 0.5243 | **REFUSE — catches it** |

Measured: intent 0.4993 @ 13.750 GiB, emitted 1.3533 @ 12.792 GiB — **2.71x worse error for 7%
fewer bytes**. Only the third assertion fires, and it fires because it measures the property we care
about (the collapse reproduced the allocation's quality) rather than a proxy for it (bytes spent).

**Requirement, more than an assertion:** `emitted_err` must be computed from the **same objective**
as the allocator used (`w * err / energy`, edge-weighted, unseen→0). Both of us computed that
objective wrongly at least once during this analysis and produced a plausible wrong answer each
time — one printed the collapse as *better* (0.651x) because a per-layer helper was called once per
expert over an all-expert sum.

| | proxy | vs uniform |
|---|---:|---:|
| per-expert intent (not expressible) | 0.507 | −54.7% |
| per-layer (expressible today) | 0.880 | −21.4% |
| collapsed (ships today) | 1.355 | **+21.0%** |

Per-layer captures **39%** of the theoretical per-expert gain; **61% is lost to the format**. The
honest question for the morning is therefore two-part: whether a fixed collapse rule recovers enough
on its own, and — separately — whether per-expert allocation is worth having **even if it could be
expressed**.

**The decision, given the mechanism.** V4 re-tiered 41.5% of experts per-expert; per-layer
granularity would be a departure from that, even though it produces the better artifact. But the
measured defect is in the *emitter's collapse rule*, not in the granularity, so the cheapest fix
keeps the V4-faithful measurement intact:

**Options:**

- **(a) ship as-is** — the emitted file is **20.9% worse than the best uniform config at the same
  bytes**. Not neutral: it spends the screen and the quantize on an allocation the format discards,
  and the vote actively picks the worst tier.
- **(a′) replace mode with a budget-aware tier choice** — 1.3533 → **0.8805** (35% error reduction)
  at essentially the same bytes, marginally better than the best uniform. **Keeps V4's per-expert
  measurement intact**, no format change, no methodology departure. Note the honest scope: the rule
  *is* the per-layer knapsack, so this is not literally one line — but it lives in the emitter's
  collapse step, not in the allocator.
- **(b) per-layer as a first-class decision** — move the knapsack into `v41_alloc.py`. Same answer,
  larger change, and it *is* a method departure from V4's per-expert re-tiering.
- **(c) keep per-expert, accept the GGUF cannot express it** — 61% of the theoretical gain never
  reaches the artifact.

**(a′) dominates (b)** on effort and on fidelity to V4, so it is the recommendation. (b) is only
worth doing if a 48-layer run shows per-layer materially ahead of a budget-aware rule — which the
analysis above says it will not, since they are the same computation.

**The per-expert *calibration* survives the collapse even though the per-expert *types* do not.**
This is a fork
feature, not upstream: `llama-quant.cpp:1278` quantises each expert separately because they have
different importance matrices —

```cpp
for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
    const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;
```

— and `llama-imatrix` writes the matching per-expert format (`.in_sum2` of shape `(ne0, n_expert)`
plus `.counts`, re-keyed to the base tensor name by `common/imatrix-loader.cpp:116-175`). Verified
on the real 100-chunk imatrix: **all 144 expert tensors (48 x gate/up/down) present, zero missing**,
and the `imatrix.size() == ne[0]*ne[2]` guard at `llama-quant.cpp:1235` passes exactly
(2560 x 512 = 1,310,720).

So: **one type, 512 calibrations.** The per-expert measurement still reaches the quantiser — each
expert's scaling is fitted to its own routed tokens — and only the per-expert *type* choice is
lost. Do not describe the collapse as "one averaged imatrix"; and the granularity A/B's expected
gain is now only the type-choice part, which is the likely reason if it comes out small.

This also makes the abort path safe: `tensor_requires_imatrix` (`llama-quant.cpp:803`) returns
true for IQ3_XXS/IQ2_XXS/IQ2_XS/IQ2_S/IQ1_*, and a missing entry **throws** ("The result will be
garbage, so bailing out") rather than degrading. Every tier our allocator can emit is covered.

**Per-expert tiers are expressed two ways, and both are already implemented for ds4:**

1. **GGUF mixed layout** (§4.0) — `ffn_<role>_exps.<type>.weight` per type + `I32
   ffn_exps_{gu,down}_type[384]`. This is what **V4 shipped**. Port `tools/v41/v41_gguf.py`.
2. **Strata native pack** (`tools/iq_pack.py` → `experts.bin`, one blob per expert) — carries a
   per-layer `native_experts.txt` today; would need extending to per-expert types.

Option 1 is the proven route and needs no format invention.

**What XQ still buys here:** replacing a depth heuristic with a measured, error-driven choice per
layer and projection, over a 7-step ladder instead of 5. On the shipped SC117 file the gate bpw
follows a crude depth ramp (2.31 for layers 0-15, 2.80 for 16-31, 2.99 for 32-47) and the down
format is chosen in a pattern that does not track it at all. That is the headroom.

**How MUCH it buys — measured, and much less than the pitch implies.** On L1–L15 with real
screening stats, comparing at **equal bytes** (9.33 GiB, what a uniform IQ2_XXS actually costs):

| allocation | bytes | proxy error |
|---|---:|---:|
| uniform IQ2_XXS (gu iq2_xxs, dn q2_0) | 9.33 GiB | 8.7984 |
| XQ per-expert allocation | 9.33 GiB | **8.6379** |

**Advantage: 1.82% of proxy error.** Not 16x, not the −96.6% / −40.6% the allocator prints — those
are versus `all-Q2_0`, a terrible baseline on a non-monotonic ladder, at *different* budgets.
Three "improvement" figures exist for the same run and only the equal-byte one is a like-for-like
comparison. That is the exact number-whose-denominator-is-missing class this file warns about.

Three reasons not to over-read 1.8% in either direction:

1. **Proxy error is not quality.** It is a sum of per-expert, energy-normalised squared output
   error — not perplexity, not top-1. A 1.8% proxy gain could be 4% or 0.5% real. The gate is head
   perplexity and top-1 against Q8_0, and it has not been run.
2. **Early layers only.** L1–L15, whose relative error spans 1.65x; L16–L47 are unmeasured.
3. **Uniform IQ2_XXS is not a shipped option** for this model. The competitive bar is UD-Q4_K_XL
   at 77 GiB / 94.1% top-1, not a hypothetical uniform IQ2 at 9.33 GiB.

If 1.8% holds at 48 layers, XQ's value here is "a defensible tier choice at a size the
alternatives do not hit", which is a weaker claim than "much better experts" — worth knowing before
quantising rather than after.

### 3.7 Screening cost — MEASURED, and it sets the wall clock

`ggml_quantize_chunk` is **single-threaded**, and `OMP_NUM_THREADS` does **nothing** for it.
Same `(640, 2560)` matrix, per tier, at OMP 1 / 2 / 4 / 8 — identical to three decimals. Full
ladder measured warm, one matrix each:

| tier | bpw | ms per (640,2560) | share of ladder |
|---|---:|---:|---:|
| Q2_0 | 2.25 | 8 | 0.2% |
| IQ2_XXS | 2.0625 | 685 | 16.5% |
| IQ2_XS | 2.3125 | **1355** | **32.7%** |
| IQ2_S | 2.5 | 372 | 9.0% |
| IQ3_XXS | 3.0625 | 803 | 19.4% |
| IQ3_S | 3.4375 | 567 | 13.7% |
| IQ4_NL | 4.5 | 357 | 8.6% |
| **sum** | | **4147** | |

**IQ2_XS costs a third of the entire ladder** for 0.0625 bpw more than Q2_0 — it is the single
biggest line item and it buys the least. It is also the obvious lever if screening ever needs to
be faster (but see the warning below: shortening `--tiers` alone is unsafe).

Per expert, end-to-end (measured 12.31 s wall, 705 MB peak RSS): quantize 8.66 s (gate/up = 7
tiers x 2 matrices = 8.29 s, down = 2 tiers = 0.36 s), matmul 0.96 s, FP8 weight load 1.85 s,
total **11.47 s**. So **one layer is ~98 min and all 48 are ~78 h single-threaded** for 24,576
experts.

Parallelism must therefore be **process-level**, not thread-level. But **this box is 8 cores /
16 SMT threads** (`nproc` reports 16; `lscpu`: "AMD Ryzen 7 5700X", 8 Core(s), 2 Thread(s) per
core), so 16 processes do **not** give 16x. Speedups measured two independent ways, 24,576
experts total against 78.3 h single-threaded:

| concurrent processes | speedup | wall clock |
|---:|---:|---:|
| 6 | 5.07x | ~15.4 h |
| 8 | 6.11–8.13x | ~9.6–12.8 h |
| **16** (OMP=1, 3 layers each) | **7.05–8.87x** | **~8.8–11.1 h** |

Both harnesses bracket the same place; the sub-16x ceiling is SMT, not a bug. Take the honest
conclusion — **screening is a ~9–11 h job, not a 5 h one.** Starting ~23:50 it ends
**~08:50–11:00**, and the ~2 h quantize then lands **~11:00–13:00**.

If that is too late, the only levers are methodological, not mechanical. **Both were evaluated and
rejected for this run**, because the brief was "maintain V4's methodology":

**(a) Sample experts.** V4's tooling has the knob — `v41_calib.py:681`,
`--expert-stride K`, "screen every K-th expert only", default 1. But the **real V4 run used
stride 1**: `quant-1017-v4/stats/L00.json` has 384 experts, ids `0..383` contiguous,
`unused_experts: 0`, and DS4 has 384 experts total — i.e. **V4 screened 384 of 384**. Sampling
would therefore be a departure, not a V4 option exercised. It would also need random rather than
contiguous indices (router norms are strongly non-exchangeable), which needs a `screen.py` change.

**(b) Shorten the ladder.** IQ2_XS alone is 32.7% of screening cost. But `v41_alloc.py:71` reads
`s.get(key + t, 0.0)` — a tier that was never screened reports **zero error**, so the allocator
would treat it as free and pick it. Dropping a tier therefore requires editing
`ARCH["qwen4exp"]["tiers"]` in `v41_alloc.py`, a file another lane owns.

### The two real deviations from V4, stated plainly

| | V4 (measured) | this run | note |
|---|---:|---:|---|
| experts screened | 384 / 384 | 512 / 512 | same coverage |
| tokens | 1,638,400 | 49,152 | **33x fewer** — dump-space bound |
| tiers | 3 (`ptq1, q2_0, q3_k`) | 7 | 2.33x more screening work |
| per-layer time | 83.5 min (GPU) | ~98 min (CPU) | 40 layers = 55.7 h for V4 |

**The token count is the largest deviation in the whole pipeline, not the expert sampling** — it is
bounded by dump size, not by choice. Each token costs `48 layers x 10,280 B` = 493 KB of dump
(`2560 f32` activations + `10 i32` ids per layer):

| tokens | dump size | fits 39 GB free? |
|---:|---:|---|
| **49,152** (used) | **24.3 GB** | yes, ~15 GB spare |
| 60,798 | 30.0 GB | hard ceiling at ~30 GB usable |
| 147,456 (3x) | 72.8 GB | **no** |
| 1,638,400 (V4) | **808 GB** | **no** |

So 49,152 is near the achievable ceiling and a 3x improvement would need a second drive. Splitting
into two dump/screen passes does **not** help: the patch writes *all 48 layers* per chunk with no
layer filter, so a second pass over different chunks yields the same 49,152 total — merging two
partial screens would need weighted error combination that `screen.py` does not implement.

The tier count is why our per-layer cost is comparable to V4's despite 33x fewer tokens.

Use 16 processes with `OMP_NUM_THREADS=1` — OMP buys nothing here and only steals cores from the
extra processes. Memory is a non-issue: 705 MB x 16 = 11.3 GB.

**Do not shorten the ladder to save time.** `v41_alloc.py:71` reads `s.get(key + t, 0.0)`, so a
tier that was never screened reports **zero error** and the allocator will treat it as free and
pick it. Restricting `--tiers` requires also restricting the allocator's `ARCH` ladder, in the
same change.

---

## 5. Abliteration — same workflow, one simplification

**Yes, the workflow is the same in every way that matters**, and it gets *simpler* here rather
than harder. What already exists in `~/AI/quetza-workspace/flashnext-ablit/`: exact directional
ablation `B*A = -lam * u u^T W`, lam = 1.50 on all 145 residual-writing targets, rank 1 except two
rank-33 subspaces on `blk.47`, `A` recomputed against our own dequantised trunk, verified
`cos = 1.000000`.

| | DS4 (as built) | Qwen (this pipeline) |
|---|---|---|
| Order | edit **before** calibration | identical — edit before screening |
| What it sees | fp32 weights, before GPTQ | F16/Q8_0 weights, before `llama-quantize` |
| Mechanism | manifest npz (`V41_ABLATE`), runtime projection in 3 hooks | adapter bake (`ablit-merge.cpp`) |
| Bake style | never writes an edited checkpoint | re-quantises in place to the original type |

The one meaningful simplification: `ablit-merge.cpp` exists to patch an **already-quantised
third-party file** in place at a fixed byte range. When we build our own quant from the F16
intermediate we can apply the edit to the **float** weights before quantisation, which is strictly
better — no requantisation leakage, no byte-range constraint, no `fix_q2_0.py` repair pass. So use
the adapter's math (`reexport_adapter.py` + the verified `B`), not its merge machinery.

**Why the order cannot change:** 48 of the 145 abliteration targets are `ffn_down_exps` — the
routed experts. Screening those experts before the edit measures a model we will not ship, and the
allocator would then spend tiers on the un-abliterated error pattern. Same rule as
`private/docs/V41_ABLITERATION_PIPELINE.md:76-80`.

Residual refusals stay as they are, by decision: the measured trade was refusals for capability,
and chasing the last few is where the published Flash-Next attempts paid KL for nothing.

---

## 5.4 Alpha pre-flight — MEASURED 2026-10-27 on real weights

`alpha_preflight.py` on layers 1, 9, 20 (every layer whose down shards had downloaded), 2 experts
each, real FP8 weights x the real adapter:

| tier | bytes/row | alpha | alpha_eff | **signal%** | **base_err** | verdict |
|---|---:|---:|---:|---:|---:|---|
| Q2_0 | 180 | 1.03 | 0.96 | 1.2 | **0.725** | **unusable** |
| IQ4_NL | 360 | 0.999 | 0.998 | **17.3** | 0.078 | edit buried in noise |
| NVFP4 | 360 | 0.993 | 0.984 | 14.6 | 0.116 | worse than IQ4_NL |
| MXFP4 | 340 | 0.996 | 0.981 | 14.6 | 0.118 | worse than IQ4_NL |
| Q8_0 | 680 | 1.000 | 1.000 | **94.6** | **0.0057** | edit stored cleanly |

- **signal%** = of the change the quantiser actually makes, how much is the edit rather than
  requantisation noise.  **base_err** = how much the format damages the tensor with no edit at all.
  `alpha_eff` alone is not enough: Q2_0 scores 0.96 while wrecking the tensor, because a large
  random noise projects onto `d_intended` by chance.
- **Validation, two independent numbers:** the edit measures **3.056%** of `||W||` (the published
  bake reports ~3%), and IQ4_NL's signal measures **17.3%** (the published table reports 17.5%).
  The tool is measuring the right thing.

**Q2_0 is not a quality format.** `quantize_row_q2_0_ref` (`ggml-quants.c:74`) sets `d = amax` with
no scale search, and because `d >= |w|` always, **code 3 is unreachable** — the effective levels are
`{-d, 0, +d}`, i.e. ternary with a bad scale. On Gaussian expert weights that is 0.72 relative
error. DeepSeek's XQ Q2_0 is a different animal (GPTQ + 11-point clip-ratio scale search); ggml's is
not. Anyone comparing our Q2_0 tiers to stock ggml Q2_0 is comparing different quantisers.

**NVFP4 does not help**, despite being the same size: E2M1 has a single mantissa bit and fits these
weights worse (0.116 vs 0.078) and carries less of the edit (14.6% vs 17.3%).

**Consequence for the expert-down decision.** Q2_0 is out, so the real choice is IQ4_NL vs Q8_0:

| | expert-down bytes | delta vs IQ4_NL |
|---|---:|---:|
| IQ4_NL | 22.65 GB | — |
| Q8_0 | 42.78 GB | **+20.13 GB** |

(+31.46 GB is Q8_0 vs Q2_0, but Q2_0 is unusable, so it is not the relevant comparison.)

**The cheapest GB in the whole plan:** the non-expert edited tensors are small. Attention
(2.01B params), SSM (0.52B) and the shared expert all fit at Q8_0 for **~+0.9 GiB** over their
current 5-6 bpw. Since the published MoE result is that the *joint* intervention does the work
(0.776 jointly vs 0.039/0.016/0.148 alone), making the dense half faithful for under 1 GiB is
better value per byte than any expert-side spend.

Cross-check from the community (`Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata`) — they
reached the same conclusion independently: "our NVFP4 GPTQ experts, **and Q8_0 in the layers that
move the answers most**", measured 95.0% top-1 vs Q8_0 against 93.6% for all-NVFP4.

## 5.5 PLE: IQ4_NL everywhere, the FP8 table is not needed (decision)

The checkpoint's PLE table is F8_E4M3 (49.6 GB across 31 shards). **We do not download it.**
The IQ4_NL table (28.80 GB) is used for calibration *and* for serving. Reasons, in order:

1. **The engine only reads IQ4_NL** (`ngram.cpp`), so the shipped artifact must be IQ4_NL either
   way. Downloading FP8 would buy a more accurate *teacher* and nothing else.
2. **Calibration must see the model we ship.** Fitting the allocator against an F8-PLE teacher
   while serving IQ4_NL would mean the per-expert errors describe a different model — the exact
   anti-pattern `V41_ABLITERATION_PIPELINE.md:76-80` warns about.
3. **The decision is relative, not absolute.** Every candidate tier for a given expert sees the
   same PLE error, so it cancels in the tier ranking.

Consequence to keep in mind: the head-perplexity gate is then measured against an IQ4_NL-PLE
reference, not true FP16. That is the right reference for a shipped IQ4_NL model, but it must not
be compared to a published FP16 number.

## 6. Open work

- [ ] **HIGHEST PRIORITY — pre-registered: does proxy error rank tiers the way head perplexity
      does?** Every allocation number in §4 is conditional on this, and it has never been tested.
      **Do not use a wide-separation pair** (e.g. `q2_0` vs `iq4_nl`, 102x apart): that would pass
      trivially and only confirm "more bits is better", which nobody doubts. Test the **single
      non-monotonic pair on this ladder**, verified as the *only* inversion in the summed
      `egu_` proxy:

      | tier | bpw | Σ proxy |
      |---|---:|---:|
      | `iq2_xxs` | **2.0625** | **4,463.9** |
      | `q2_0` | **2.2500** | **23,000.4** |

      `iq2_xxs` is **cheaper by 0.1875 bpw and 5.15x better**, predicted by nothing except the
      proxy — not bit count, not a "more bits is safer" prior.

      **Pre-registered prediction: quantize one layer's experts at `q2_0` and at `iq2_xxs`;
      `iq2_xxs` must give LOWER head perplexity.**

      Both branches decide something real:
      - **passes** → proxy ranking transfers *and* the `--unseen-floor iq2_xxs` change (which moved
        168 experts onto exactly this tier) is justified;
      - **fails** → proxy error mis-ranks precisely where we trusted it, the unseen-floor change was
        wrong, and every allocation number in §4 is precise and irrelevant.

      **A failure would be LOCAL, not global — which says where to look.** The proxy behaves exactly
      as a quality measure should on every pair but one:

      | ladder | pairs | inversions |
      |---|---|---:|
      | gate/up (7 tiers, 2.06–4.50 bpw) | 6 adjacent | **1** (`iq2_xxs`→`q2_0`) |
      | **down (2 tiers, 108.2x error span)** | 1 | **0** |

      So a failed test means the defect is in the **scalar-vs-lattice comparison specifically**,
      not in proxy error as a measure: `q2_0` is a plain scalar quantiser with no codebook while
      `iq2_xxs` is lattice-based, and the proxy scores energy-normalised squared output error over
      routed rows — which need not capture whatever structure the lattice preserves. A *global*
      failure would require the monotonic down ladder to have agreed by luck, which is far less
      likely.

      The inversion is also **structural rather than an aggregation artifact**: `q2_0` scores worse
      than `iq2_xxs` in **15 of 15 layers**, not merely in the sum.

      **But the ratio's constancy is an AGGREGATE property, not a per-element one — measured.**
      Per *expert*, over 7,504 experts, the ratio spreads **32%** (median 5.13, p5 4.58, p95 6.07,
      p95/p5 = 1.32x). So a single element's penalty varies substantially even though the layer
      total does not.

      **Name the aggregation — two different statistics have both been called "the per-layer ratio"
      in this analysis, and they disagree by ~50% in spread:**

      | aggregation | mean | sd | span | LLN excess |
      |---|---:|---:|---:|---:|
      | **ratio of summed errors** (`Σq2_0 / Σiq2_xxs`) | 5.162 | 0.195 | 16.8% | **8.4x** |
      | **mean of per-expert ratios** | 5.211 | 0.134 | 9.4% | **5.8x** |

      Use **ratio-of-sums** for the aggregate claim ("this layer's total error is 5.16x") and
      **mean-of-ratios** for anything about spread or trend — the latter weights each expert
      equally instead of letting high-error experts dominate. They answer different questions, and
      mixing them is the same-objective clause again, one level down.

      **The conclusion survives both** (LLN predicts sd ≈ `0.521/√500` = 0.023; observed 0.195 or
      0.134, i.e. 8.4x or 5.8x — well above 1x either way): the layer constant carries **real
      per-layer structure beyond averaging**, plus a genuine depth trend. Under mean-of-ratios:
      5.53 at L01 → ~5.04 at L11 → 5.33 at L15.

      Why the aggregation choice matters for the test: **L05 looks like an outlier under
      ratio-of-sums (4.83) and is unremarkable under mean-of-ratios (5.23)** — that "outlier" is an
      artifact. Testing perplexity against the wrong target would produce a spurious mismatch.

      **The test is now three-outcome, not two**, because the layer means are *not* pure LLN
      smoothing: they carry ~6–8x more spread than averaging can produce, plus a real depth trend.
      So:

      1. perplexity margin **tracks the depth trend** (high L01, minimum mid-depth, upturn L15) →
         strong evidence the proxy transfers at layer level;
      2. margin **flat** → uninformative (a flat margin is what averaging alone would give);
      3. margin **layer-dependent but in a *different* pattern** → the two measure different things.

      Outcome 1 is a far stronger test than "is it flat", and it is only available because the
      trend exists.

      **A sharper test the spread now makes possible (later):** quantize one layer with `q2_0`
      applied only to its top-quartile-ratio experts, then only to its bottom-quartile. If the
      damage tracks the per-expert ratio, the proxy is valid per-expert — which is exactly the
      condition under which per-expert allocation could ever pay off, answering the §4 reframe
      empirically. If both quartiles damage equally, the expert-level variation is noise and only
      the aggregate is meaningful.

      **Coherence check that passes:** the inversion is on gate/up only, and the unseen-floor change
      was surgical to gate/up for exactly that reason — down keeps `[q2_0, iq4_nl]` under either
      floor, because both exceed 2.0625. The one thing changed tonight sits precisely on the one
      inversion, and the ladder where the change does nothing is the ladder with no inversion.

      Until this runs, §4's percentages should be quoted only with the caveat that the ranking is
      unvalidated.
- [x] Converter handling for the missing PLE tensors (§3.1) — `filter_index.py` +
      `convert_ple_deferred.py`, both tested; the vendored tree is untouched.
- [ ] First load with `--ple-gguf` on a real GGUF, to confirm the engine accepts PLE metadata
      without the `per_layer_token_embd` tensor.
- [x] **Screener** (`screen.py` + `ggml_qshim.cpp`) — written and validated on a synthetic
      checkpoint. It does **not** need Qwen's Python: the teacher activations come from the engine's
      own MoE dumps, so `transformers` support for `qwen4_exp` is irrelevant.
- [x] **Screener validated end-to-end on REAL FP8 weights (2026-10-27)** — synthetic dumps in the
      engine's exact format (N=2560, K=10), real checkpoint, `--limit-expert 0`:
      `FP8Model` reads the 151,965-entry index in 0.3 s; L0/e0 geometry gate(640,2560)
      up(640,2560) down(2560,640) matches `ARCH["qwen4exp"]`; 1536 expert tensors per layer
      (512 x 3); down ladder correctly restricted to `[q2_0, iq4_nl]`; tier error strictly
      monotone (q2_0 8.56e-2 > iq2_xxs 1.99e-2 > iq2_xs 1.48e-2 > iq2_s 1.04e-2 > iq3_xxs 5.59e-3 >
      iq3_s 3.39e-3 > iq4_nl 9.59e-4). Independent instrument cross-check: the shim's
      quantize->dequantize rel_err on 2560-wide rows is q2_0 0.712 / iq2_xxs 0.342 / iq3_s 0.137 /
      iq4_nl 0.075, which reproduces `alpha_preflight.py`'s real-weight 0.725 / 0.078. Both
      held-out branches exercised (n_fit=22: `--min-fit 32` -> `n_im=0, lowdata`; `--min-fit 16`
      -> `n_im=22`). The only untested link left is the engine's own dump payload.

      **Dump header shape is `(prefill, 2560, 10)` — T is the chunk size, not a constant.** T is
      `std::min(m.T, n - c0)` (`prefill.cpp:2199`) and `N`/`K` are the compile-time constants at
      `prefill.cpp:118`, so with `--prefill 2048` every file is `T=2048` and exactly
      `24 + 2048*2560*4 + 2048*10*4 = 21,053,464` bytes; 48 layers x 24 chunks = 24,253,590,528 B
      (24.25 GB). Verified by running screen.py at `T=2048` as well (`n_fit=30 n_eval=9`).
      Failure modes measured on the consumer side: a **wrong `N` fails loudly** —
      `ValueError: matmul: ... (size 2560 is different from 1000)`, exit 1 — but a wrong `T` would
      be consumed silently (`read_dump` trusts the header), so the dump must be header-checked
      before screening, not validated by screen.py's success.
- [ ] **OPEN: expert routing is heavily imbalanced — MEASURED 2026-10-27, and it changes how the
      stats must be read.** The gate is top-k over raw logits (`output_gate_type=sigmoid`, no
      cross-expert normalisation, and no `e_score_correction_bias` tensor exists in the
      checkpoint), so each expert's logit sd is proportional to its router row norm. Measured
      row-norm spread (max/median) by layer: L0 3.53x, L1 2.05x, L12 2.13x, L24 2.45x, L36 3.96x,
      L47 2.58x. On L0, `corr(row norm, usage) = 0.965`; the top-10-norm experts carry 19.1% of
      assignments and the bottom-100 carry 1.0%; max/mean = 13.9x and some experts get **zero**.
      Config corroborates: `router_aux_loss_coef = 0.001` is almost no balancing pressure.

      **MEASURED from the real dump (2026-10-27), which supersedes every simulation below.**
      Counted by replaying `screen.py`'s own `RandomState(1234+layer).rand(49152) < 0.25` and the
      actual routed rows across all 24 chunks, so `n_fit` is exactly what the screener computes:

      | layer | distinct | unrouted | **lowdata (`n_fit<32`)** | % | max/mean | isotropic model said |
      |---|---:|---:|---:|---:|---:|---:|
      | 0 | 503/512 | 9 | **30** | 5.9% | 5.88x | 48 (9.4%) |
      | 12 | 512/512 | 0 | **8** | 1.6% | 8.70x | 0 |
      | 24 | 512/512 | 0 | **10** | 2.0% | 7.89x | 12 (2.3%) |
      | 36 | 511/512 | 1 | **57** | 11.1% | 24.67x | 8 (1.6%) |
      | 47 | 508/512 | 4 | **77** | 15.0% | 37.84x | 76 (14.8%) |

      The isotropic-input model got the *shape* right and two layers nearly exact, but **badly
      under-predicted depth**: real max/mean is 24.67x at L36 and 37.84x at L47, against predicted
      16.5x and 11.3x, so L36 was off by 7x (8 predicted, 57 measured). Aggregate is ~7% of
      experts, not the 9.4% quoted for L00 — so the q2_0 floor costs **~4.5 GB, not ~1.3 GB**.

      (History, kept because the reasoning was wrong twice: an early figure of 90–130 came from a
      16,384-token simulation quoted as if it were the 49,152 budget; the corrected estimate was
      "L00 40–60, L47 70–90". Both are now superseded by the measurement above.)

      These experts take the uniform imatrix and the `--unseen-floor q2_0` guard, but **on
      L36/L47 (11–15%) the tier histogram partly measures corpus coverage rather than
      quantisation error** — L00/L12/L24 (1.6–5.9%) are well measured. Report lowdata% beside the
      tier histogram and the distinction is quantified rather than assumed.

      **Dump sanity — two different units, do not mix them.** Measured on real routers, healthy:

      | unit | distinct | zero-count | max/mean |
      |---|---|---:|---:|
      | one 2048-token chunk (20,480 ids) | 450–512 | 0–62 | 8.4–17.6x |
      | full budget (49,152 tokens) | 494–498 | 14–18 | 11.3–13.5x |

      A per-chunk check therefore reports **~40 unselected experts and ~450–471 distinct**, not the
      full-budget 18/494. Both are healthy. The three warning directions, with their correct sign:
      a *flat* warning must trigger on `max/mean < 3` (nothing this router emits is that flat — it
      can only come from ids that are not this router's output), so raising such a threshold to
      20x would invert it and fire on every healthy layer, whose max is 8.4–17.6x. Degenerate is
      `<400` distinct or **~0 lowdata**. Raising the token budget is the only
      real fix and it is dump-space bound (48 layers x total_tokens x 10,280 B).
- [x] **Engine link for `strata` restored (2026-10-27).** `build-xolotl/strata` did not link —
      20 undefined refs to the 3-arg `ggml_backend_cuda_context::new_pool_for_device` from
      `libstrata_mmq.a`. The vendored ggml-cuda gained that signature; `src/prefill/ggml_cuda_host.cu`
      still had the 2-arg shim only. A forwarding definition in the shim fixed it (the build sets
      `vmm = false`, so the 3-arg form's only reachable branch is the one the 2-arg already takes).
      Verified: control build without the change fails identically, so it was pre-existing.
- [ ] **Apply + build `dump_moe_all.patch`**, then validate on one layer: dump L0, screen it, check
      the tier ordering matches the synthetic run. The patch applies cleanly and `prefill.cpp`
      compiles; what blocked the first attempt was the link, now fixed.
- [ ] **Run the granularity A/B on 2–3 layers** once stats exist: compare per-expert tiers against
      the per-layer collapse. DS4 re-tiered 41.5% of experts; if Qwen is similar, per-expert
      storage is justified.
- [ ] **Per-expert storage** (only if the A/B says it matters): `native_experts.txt` v5 with
      per-expert type/offset tables, `ExpertLayout` per-expert `NativeFmt`, and the 187
      `blob_bytes()` call sites. Neither a stock GGUF nor today's native pack can express it.
- [ ] Decide whether to extend the pack format to per-expert types (§4).
- [ ] Gate: head perplexity + top-1 against the F16 teacher, and a capability panel, before
      shipping.

## 7. Published quant ladder for this model (context for the quality gate)

`Maximilian228` measured top-1 agreement against a full Q8_0 of the same model, on RAM:

| quant | top-1 vs Q8_0 | decode | 32K prompt | RAM |
|---|---:|---:|---:|---:|
| Q8_0 (all experts) | 100% (ref) | 47 t/s | 10.3 s | 106 GiB |
| NVFP4 + Q8_0 mix | 95.0% | 91 | 4.2 s | 105 GiB |
| Unsloth UD-Q4_K_XL | 94.1% | 169 | 4.9 s | 77 GiB |
| NVFP4 GPTQ (all experts) | 93.6% | 177 | 3.9 s | 69 GiB |
| NVIDIA NVFP4 | 92.9% | 167 | 3.9 s | 69 GiB |
| NVIDIA FP4-Local (QAD) | 92.2% | 191 | 3.7 s | 69 GiB |
| Unsloth UD-Q3_K_XL | 90.7% | 208 | 3.8 s | 58 GiB |
| ISTA-DASLab IQ3_S | 90.0% | 228 | 3.9 s | 52 GiB |

Two uses: (1) the bar our XQ has to clear is **UD-Q4_K_XL's 94.1% at 77 GiB** — below that, all the
machinery has not paid for itself; (2) it bounds what low-bit costs on this model, and the current
served SC117 IQ3_XXS sits in the low-90s by this scale.

Note this is a *different* metric from `docs/QUANTS.md`'s KL-vs-Q8_0 table — top-1 agreement is
coarser. Do not mix the two.

## 5.5b Dense edited families — MEASURED (layers 1, 3, 20)

Same pre-flight, the non-expert abliteration targets:

| family | tensors | params | IQ4_NL signal% | Q8_0 signal% | Q8_0 base_err |
|---|---:|---:|---:|---:|---:|
| `self_attn.o_proj` (attn) | 12 | 0.189B | pending shard | pending | — |
| `linear_attn.out_proj` (ssm) | 36 | 0.566B | 18.8 | 94.6 | 0.0058 |
| `shared_expert.down_proj` | 48 | 0.079B | 17.2–20.9 | 94.0–95.5 | 0.0060 |
| `ple.value_proj` | 1 | 0.007B | pending shard | pending | — |
| **all dense edited** | **97** | **0.840B** | ~18 | ~94.5 | ~0.006 |

The dense half behaves exactly like the experts: IQ4_NL carries ~18% of the edit, Q8_0 carries ~95%.

**Cost of making the whole dense half faithful: 0.89 GB at Q8_0, i.e. +0.42 GB over IQ4_NL.**
(Earlier notes said "+0.9 GiB" — that was the absolute size, not the delta. The delta is 0.42 GB,
or ~0.31 GB against a 5.5 bpw baseline.)

That is the cheapest quality-per-byte in the pipeline, and the MoE literature says the dense and
attention writers are part of what makes the joint edit work at all (0.776 jointly vs 0.039 alone
for attention, 0.016 alone for dense). Recommendation stands: **Q8_0 the entire dense edited set
before spending anything on the expert side.**

## 5.5c Total file impact of the abliteration-faithful choices

| component | params | IQ4_NL | Q8_0 | delta |
|---|---:|---:|---:|---:|
| dense edited (attn+ssm+shexp+ple) | 0.840B | 0.47 GB | 0.89 GB | **+0.42** |
| expert down (48 x 512) | 40.265B | 22.65 GB | 42.78 GB | **+20.13** |
| **total** | | **23.12 GB** | **43.67 GB** | **+20.55** |

The dense half costs 2% of what the expert half costs for the same "faithful edit" property.
