# Strata — operational howto

Two paths, two binaries, two model families. They are not interchangeable.

| | MoE | Dense |
|---|---|---|
| Binary | `engine/strata` | `build-dense/strata-dense` |
| Models | Qwen3.8-Flash-Next (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S), Coder IQ1_M | Qwen3.8-27B (unsloth GGUF, any quant) |
| Architecture | `qwen4exp`, 48 layers × 512 experts, 2560 wide | `qwen35`, 64 layers, n_embd 5120, n_ff 17408 |
| Expert cache / profile | yes | **no** (no experts at all) |
| Speculative decoding | yes (MTP + suffix drafter) | no |
| Pack required | yes (`tools/strata_pack.py`) | no, it reads the GGUF as is |

`strata` rejects a dense GGUF at the first guard of `check_architecture`; `strata-dense` has no idea what an expert is.

---

## 1. Expert profile — what it is and why it decides the speed

### The problem

The model has **48 × 512 = 24576 experts**. The router picks **10 per layer per token**. The blobs do not all fit in VRAM.

The profile (`expert-profile.bin`) is a **static residency plan**: an ordered list of `(layer, expert)` pairs. The engine takes the first N and preloads them into VRAM.

- **hit** → resident expert, computed on the GPU (cost ~0)
- **miss** → computed on the CPU pool (~57 µs each, see below) or streamed over PCIe

### File format

Defined in [`include/strata/core/expert_cache.hpp:43`](../../include/strata/core/expert_cache.hpp:43), read by [`src/core/expert_cache.cpp:12`](../../src/core/expert_cache.cpp:12):

```
offset  0  "STRP"                        magic, 4 bytes
offset  4  version, n_layers, n_expert,
           slots, n_ranked               5 × uint32 little-endian
offset 24  n_ranked × (uint16 layer, uint16 expert)   the ordered list
           n_layers × n_expert × int32                slot-per-expert table (lookup)
```

The engine uses **only the ordered list**. The lookup table is the tool's internal work. The `slots` field is informational: it is the "built for N slots" that shows up in the log.

Validation ([`src/core/expert_cache.cpp:32`](../../src/core/expert_cache.cpp:32)): if `(n_layers, n_expert)` does not match the model geometry, the engine stops with

```
read_expert_profile: <file> is 48x512 but this model is 64x256 -
it is a profile for a different artifact
```

### The ranking matters more than the slot count

MoE routing follows a power law. The repo models it like this in [`src/program/generate.cpp:1998`](../../src/program/generate.cpp:1998):

```
dispatched mass for rank r  ∝  (r+1)^-1.2
```

Exponent 1.2 < 1 → heavy tail: the few experts at the top take almost all the traffic. So **coverage by expert count ≠ coverage by dispatched mass**, and it is the mass that determines the hit rate.

### The default trap, with the numbers

[`tools/make_default_profile.py`](../../tools/make_default_profile.py) generates pairs in **layer-major** order: all 512 experts of layer 0, then layer 1, etc. Useful only as a placeholder to get the engine started.

With 9615 slots (the ones that fit on a 3090/4090 with IQ3_S):

```
9615 ÷ 512 = 18.78
9615 = 18 × 512 + 399
```

| Layer | Resident experts | Coverage |
|---|---|---|
| 0 – 17 | 512 / 512 | 100% |
| 18 | 399 / 512 | 77.9% |
| 19 – 47 | **0 / 512** | **0%** |

Expected hit rate (48 equivalent layers):

```
h = (18 + 0.779) / 48 = 39.1%
```

Measured: **43.7 – 44.4%**. The +5% comes from layer 18 getting *its own* 399 hottest experts.

The structural damage: **29 layers out of 48 with an exactly 0% hit rate**, whatever the prompt.

Counter-figure — by-frequency profile, 9615 slots spread across all layers:

```
9615 ÷ 48 = 200.3 slots per layer
```

Mass coverage for one layer, with Zipf exponent 1.2:

```
h = Σ_{r=1}^{200} r^-1.2  ÷  Σ_{r=1}^{512} r^-1.2
```

Partial sum of ζ(1.2) ≈ 5.60; tail from 513 ≈ ∫ x^-1.2 dx = 5·513^-0.2 = 1.44; tail from 201 ≈ 5·200.5^-0.2 = 1.73:

```
denominator = 5.60 − 1.44 = 4.16
numerator   = 5.60 − 1.73 = 3.87
h = 3.87 / 4.16 = 93.0%
```

Measured: **86 – 94%**.

| | layer-major | by-frequency |
|---|---|---|
| Slots | 9615 | 9615 |
| Resident experts | 39.1% | 39.1% |
| **Dispatched mass covered** | **39.1%** | **93.0%** |
| Measured hit rate | 43.7–44.4% | 86–94% |

Same slots, same number of experts in VRAM. The *right* 39% of experts carries 93% of the traffic.

### From hit rate to tok/s

Model from the costs in [`include/strata/spec/controller.hpp:25`](../../include/strata/spec/controller.hpp:25), window T=4 (n = k+1 = 4):

```
step_ms = dense(n) + experts(n) + sync + draft

dense   = 11.0 × dense_ratio[3] = 11.0 × 1.45              = 15.95 ms
experts = 15.8 × distinct_ratio[3] × (1 − h) = 15.8 × 2.88 × (1 − h)
sync    = 2.4 ms
draft   = 3 × 1.2 = 3.6 ms
```

| h | experts | step_ms |
|---|---|---|
| 0.90 | 4.55 | **26.5** |
| 0.44 | 25.48 | **47.4** |

Expected ratio: `47.4 / 26.5 = 1.79×`.

Empirical, same model and same machine, two checkouts:

| Run | h | tokens | ms | ms/token |
|---|---|---|---|---|
| engine 0.1.27 | 86.0% | 31047 | 398471 | **12.83** |
| engine 0.1.30 | 43.8% | 1420 | 36530 | **25.72** |

```
25.72 / 12.83 = 2.00×  slower
```

The 1.79 vs 2.00 gap is the non-linear drain of the CPU pool: with 308 misses/token the queue adds up and the window waits for the slowest worker, not the average.

### Real cost of a miss, derived from the logs

```
lookups/token:  779114 / 1420   = 548.7   (h = 44%)
                18414313 / 31047 = 593.1  (h = 86%)
                expected ~480 = 48 layers × 10 experts

miss/token:     548.7 × 0.562 = 308.4
                593.1 × 0.140 =  83.0      → Δ = 225.4

Δms/token:      25.72 − 12.83 = 12.89

per miss = 12.89 / 225.4 = 0.057 ms = 57 µs
```

Sanity check: 480 experts all miss → 480 × 57 µs = 27.4 ms, against the CostModel's `cpu_all_miss_ms = 15.8`. The model underestimates by ~1.7× because it assumes the 19 workers are perfectly parallelized.

### Full chain

```
9615 identical slots
   → different order
   → covered mass 39.1% vs 93.0%
   → hit rate 44% vs 90%
   → misses 308 vs 83 /token
   → 25.7 vs 12.8 ms/token
   → ~45 vs ~90 tok/s
```

No bug, no code regression.

---

## 2. Building a by-frequency profile from scratch

### 2.1 Tokenize a text from your real workload

```bash
cd /home/albus/workspaces/AI/Strata/RiStarta/Strata

python3 - <<'PY'
import sys; sys.path.insert(0, "tools")
from strata_tokenizer import Tokenizer

gguf = "/home/albus/workspaces/rag_rss/models/llm_models/models/IQ3_S/" \
       "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
tk   = Tokenizer.from_gguf(gguf)
text = open("/tmp/corpus.txt", encoding="utf-8").read()
ids  = tk.encode(text, parse_special=True)
open("/tmp/tokens.txt", "w").write(" ".join(map(str, ids)))
print(len(ids), "tokens")
PY
```

`/tmp/corpus.txt` = paste code, chat, tool calls: what the model actually does. The profile is a photograph of the prompt distribution. If the corpus does not represent it, the hit rate will not go up.

### 2.2 One-shot run that dumps the trace

```bash
engine/strata \
  --pack /home/albus/workspaces/AI/Strata-data/packs/iq3_s \
  --native /home/albus/workspaces/rag_rss/models/llm_models/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --ple-gguf /home/albus/workspaces/rag_rss/models/llm_models/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --tokens-file /tmp/tokens.txt \
  --max-new 12000 \
  --max-context 32768 \
  --dump-routing /tmp/trace.bin \
  --expert-cache 0 --prefill 0 --spec 0
```

Why exactly those flags:

| Flag | Reason in the code |
|---|---|
| `--prefill 0` | **the most important one**. With prefill active the prompt travels in batch inside `prefill.run()` and never goes through `drive_pool`: zero records for the prompt tokens. Only `drive_pool` ([`generate.cpp:590`](../../src/program/generate.cpp:590)) and `drive_pool_multi` ([`generate.cpp:622`](../../src/program/generate.cpp:622)) write the trace |
| `--expert-cache 0` | disables the cache → `hit_fn = nullptr` ([`generate.cpp:2789`](../../src/program/generate.cpp:2789)). The pool sees all k experts with no hit/miss ambiguity |
| `--spec 0` | uses `drive_pool` instead of `drive_pool_multi`. The multi path writes **unit weights** ([`generate.cpp:627`](../../src/program/generate.cpp:627)), the single path writes the real router weights |
| no `--expert-profile` | the trace must not be contaminated by a previous residency |
| **not** `--no-pool` | forbidden: [`generate.cpp:2775`](../../src/program/generate.cpp:2775) exits with "needs the expert pool" |

You need a **free GPU** (~18 GiB). At the end it prints:

```
routing dumped         /tmp/trace.bin (N records of layer, k, ids, weights)
```

Record format: `int32 layer, int32 k, k int32 ids, k float weights`. Expected size:

```
(8 + 8×10) bytes × 48 layers × ~13000 tokens ≈ 66 MB
```

### 2.3 Build the profile

```bash
mkdir -p /home/albus/workspaces/AI/Strata-data/profiles

python3 tools/make_profile.py /tmp/trace.bin --no-base \
  --out /home/albus/workspaces/AI/Strata-data/profiles/iq3_s-chat.bin
```

`--no-base` is the key for "from scratch": without it, the tool puts the `--base` ranking first and your pairs end up behind ([`make_profile.py:82`](../../tools/make_profile.py:82)). With `--no-base` the order is: pairs from the trace by descending frequency, then the missing ones **interleaved across layers** ([`make_profile.py:91`](../../tools/make_profile.py:91)) — exactly what avoids the layer-major trap.

More traces are merged by summing the frequencies:

```bash
python3 tools/make_profile.py /tmp/trace-code.bin /tmp/trace-chat.bin /tmp/trace-tools.bin \
  --no-base --out /home/albus/workspaces/AI/Strata-data/profiles/iq3_s-mixed.bin
```

Pruned model (Coder: 256 experts out of 512):

```bash
python3 tools/make_profile.py /tmp/trace.bin --no-base --n-expert 256 \
  --out /home/albus/workspaces/AI/Strata-data/profiles/coder-iq1_m.bin
```

### 2.4 Point the config at the profile and measure

In the `strata-<model>.json` config, `args` field:

```json
"--expert-profile", "/home/albus/workspaces/AI/Strata-data/profiles/iq3_s-chat.bin",
"--expert-cache", "auto",
"--expert-cache-per-layer"
```

Then look for this in the log:

```
profile ...: 24576 ranked pairs, built for 24576 slots
expert cache 9615 slots, 17.29 GiB of VRAM
pre-filled 9615 of 9615 slots from the profile; slot 0 verified
decode expert cache hit rate: 8x.x%
```

### 2.5 How much trace you need

Per layer you have `10 × N_token` samples over 512 experts:

```
to rank the top-200 (the ones that fit into the 9615/48 slots):
    10N ≫ 200 · ln(512) ≈ 6200   →  N ≳ 600
for the tail (512 experts):
    10N / 512 ≥ 20               →  N ≥ 1024
recommended:
    N = 10000 – 20000            →  100k–200k samples/layer, ~200–400 per expert
```

Below ~2000 decode tokens the tail ranking is noise; above ~30000 you gain nothing.

### 2.6 Portability

| Case | Works? |
|---|---|
| Same model, different quantization (Q2_0 ↔ IQ3_S) | **yes**, reuse the same file. Routing depends on the weights, not on how they are quantized |
| Pruned model (256 experts) | **yes**, with `--n-expert 256` |
| Architecture with a different number of layers or experts | **no**: the engine rejects the file. You need a new trace and a parameterized tool (`N_LAYER, N_EXPERT` are hard-coded in [`make_profile.py:23`](../../tools/make_profile.py:23)) |
| Dense model (Qwen3.8-27B) | **irrelevant**: no experts, no profile |
| Different workload (code → chat) | the profile stays valid but performs worse: regenerate it |

Keep the profiles **outside the repo** (they are in `.gitignore`):

```
Strata-data/profiles/qwen3.8-flash-next.bin
Strata-data/profiles/coder-iq1_m.bin
```

`--expert-profile` is also mandatory for two features that otherwise will not start: layer split across multiple GPUs ([`generate.cpp:1264`](../../src/program/generate.cpp:1264)) and `--resident-cpu-experts` ([`generate.cpp:1250`](../../src/program/generate.cpp:1250)).

### 2.7 Controlling what stays resident

| Flag | Effect |
|---|---|
| `--expert-cache auto` | sizes itself from the slots the free VRAM can pay for (minus reserve, prompt buffer, drafter) |
| `--expert-cache N` | truncates the list to N: try "what 2000 slots gives you" without regenerating the file |
| `--expert-cache 0` | cache disabled, everything on the CPU pool |
| `--expert-cache-per-layer` | each layer gets **its own** slots instead of a shared counter |

The last one is decisive. From the help in [`generate.cpp:506`](../../src/program/generate.cpp:506): without per-layer, the shared counter sends the first slots all to the low layers → **2.97%** hit. With per-layer: **21.4%** at 8 slots/layer, **70.4%** at 64/layer.

The profile is only the **initial state**: the adaptive tier then swaps experts based on what the conversation actually dispatches, every `--adapt-every` round.

---

## 3. Qwen3.8-27B (dense) from the command line

### 3.1 Build

```bash
cmake -B build-dense -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DSTRATA_BUILD_TESTS=OFF
cmake --build build-dense --target xolotl-dense -j
```

Shortcut that does everything (GGUF download, tokenizer, build, config):

```bash
python3 tools/dense_setup.py --gguf /path/Qwen3.8-27B-UD-Q4_K_M.gguf --build --arch 86 --context 32768
# or let it download the model itself
python3 tools/dense_setup.py --download UD-Q4_K_M --build --arch 86
```

No pack step: `strata-dense` reads the GGUF blocks as they are, with any type the native GEMV kernels accept (Q4_K, Q5_K, Q6_K, Q3_K, Q8_0, IQ3_S, IQ4_XS, IQ2_*).

Kernel verification without a model:

```bash
build-dense/strata-dense --selftest
```

### 3.2 Available flags

From [`src/program/dense_main.cpp:254`](../../src/program/dense_main.cpp:254) — there are few, and `--serve` is **mandatory**:

```
strata-dense --serve --native <model.gguf> [--context N]
```

| Flag | Notes |
|---|---|
| `--serve` | mandatory. There is no one-shot mode: the program always speaks the protocol over stdin/stdout |
| `--native` / `--model` / `--gguf` | GGUF path (single file or first shard of a split) |
| `--context` / `-c` | context cells, default 32768 |
| `--selftest` | tests the CUDA kernels against a CPU reference, no model needed |

The flags `setup` passes to `strata` (`--gpu-layers`, `--cache`, …) are **silently ignored** here ([`dense_main.cpp:258`](../../src/program/dense_main.cpp:258)).

### 3.3 stdin/stdout protocol

Documented in [`src/program/dense_main.cpp:1`](../../src/program/dense_main.cpp:1):

```
out   INFO key=value ...                    facts for the Monitor
out   READY <context> stop                  the engine is ready
in    GEN <max_new> [key=value ...] id,id,...
        keys: temperature top_p top_k min_p
              penalty_last_n penalty_repeat penalty_freq penalty_present seed
out   PP <position> <prompt tokens> <ms> <tok/s>       every ~32 prompt tokens
out   T <id>                                          one generated token
out   DONE <generated> <prompt> <prompt_ms> <decode_ms> <stop|length|cancel> 0 0 <reuse>
in    STOP | QUIT
```

### 3.4 Manual run

```bash
GGUF=/media/albus/windows/Users/mio/models/llm_models/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf

# tokenize the question
python3 - <<'PY' > /tmp/q.txt
import sys; sys.path.insert(0, "tools")
from strata_tokenizer import Tokenizer
tk = Tokenizer.from_gguf("/media/albus/windows/Users/mio/models/llm_models/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf")
print(",".join(map(str, tk.encode("Explain in two sentences what a GPU does.", parse_special=True))))
PY

# feed GEN on stdin, read the tokens
printf 'GEN 128 temperature=0 top_k=1 %s\nQUIT\n' "$(cat /tmp/q.txt)" \
  | build-dense/strata-dense --serve --native "$GGUF" --context 4096
```

`temperature=0` = greedy ([`include/strata/kernels/sampler.hpp:19`](../../include/strata/kernels/sampler.hpp:19)).

### 3.5 Run through the server (OpenAI/Anthropic + web app)

```bash
python3 serve/server.py --engine strata --config strata-qwen3.8-27b.json --port 8080
```

Current config in [`strata-qwen3.8-27b.json`](strata-qwen3.8-27b.json): `exe` points to `build-dense/strata-dense`, `tokenizer` to `data/packs/qwen3.8-27b/tokenizer`. Terminal chat:

```bash
python3 chat.py --port 8080
```

### 3.6 VRAM budget and weights in host memory

The model computes by itself how much fits in VRAM ([`src/core/dense_model.cpp:441`](../../src/core/dense_model.cpp:441)):

```
budget = free VRAM − KV cache − 1 GiB margin
```

What does not fit in VRAM goes into **mapped pinned host memory**, and the GEMV rereads it over PCIe on every token. The log says it explicitly:

```
strata-dense: 14 tensors (11.20 GiB of 15.40) live in host memory and are read over PCIe
```

That line is the number one bottleneck of the dense path. To control it:

| Lever | Effect |
|---|---|
| shorter `--context N` | the KV cache takes less space, more weights fit into VRAM |
| `STRATA_DENSE_GPU_MIB=<MiB>` | explicit weight budget, ignores the auto-computation ([`dense_model.cpp:451`](../../src/core/dense_model.cpp:451)) |
| smaller quantization | fewer total bytes to place |

If the KV does not fit even with the margin, the engine stops with:

```
dense model: the KV cache for a context of N needs X MiB, but only Y MiB of VRAM are free
(1 GiB is kept as margin). Use a shorter --context.
```

### 3.7 Debug

```bash
STRATA_DENSE_DEBUG=0:2 build-dense/strata-dense --serve --native "$GGUF"
```

Traces the device vector statistics for steps with `0 <= position < 2`, layer by layer ([`dense_model.cpp:417`](../../src/core/dense_model.cpp:417), [`:286`](../../src/core/dense_model.cpp:286)). Useful to understand whether a layer produces NaN or the wrong scale.

### 3.8 Known limits of the dense path

- **No speculative decoding.** The loop is pure autoregressive, one token per forward pass. GDN has a recurrent state that advances one token at a time and the residual stream creates a sequential dependency: you cannot batch T tokens in a single pass.
- **Prompt read one token at a time** ([`dense_main.cpp:20`](../../src/program/dense_main.cpp:20)). Correct and comparable with llama.cpp; a batched prefill is the next optimization, not a prerequisite.
- **Conversation cache for prefixes only.** The GDN state cannot be rolled back: a request reuses the state only if everything it consumed is a prefix of the new prompt. Otherwise reset and prompt reread ([`dense_main.cpp:16`](../../src/program/dense_main.cpp:16)).

---

## 4. MoE (Qwen3.8-Flash-Next) — quick reference

```bash
# one-shot, no server
engine/strata \
  --pack /home/albus/workspaces/AI/Strata-data/packs/iq3_s \
  --native <model>-00001-of-00002.gguf \
  --ple-gguf <model>-00002-of-00002.gguf \
  --expert-profile /home/albus/workspaces/AI/Strata-data/profiles/iq3_s-chat.bin \
  --expert-cache auto --expert-cache-per-layer \
  --prefill auto --spec 4 --spec-min-p 0.5 \
  --mtp /home/albus/workspaces/AI/Strata-data/mtp/rt \
  --max-context 128000 --kv int8 --kv-resident 32768 \
  --tokens-file /tmp/tokens.txt --max-new 512
```

```bash
# server
python3 serve/server.py --engine strata --config strata-iq3_s.json --port 4332
```

Speculation: `--spec T` (T ≤ 8) enables the verify window, `--mtp <dir>` the draft layer, `--suffix-draft N` the prompt lookup (default 3, `0` turns it off). The suffix drafter has no weights and costs microseconds: it pays off on code and repeated text, +6-11% ([`generate.cpp:1358`](../../src/program/generate.cpp:1358)).

Diagnostics in every run:

```
session is up (engine 0.1.30)
expert cache 9615 slots, 17.29 GiB of VRAM
decode expert cache hit rate: 8x.x% (N hits / M lookups)
speculation   N rounds of 4, drafts accepted X of Y (0.65), 3.1 tokens per round
```

If the hit rate drops below ~80%, the number one suspect is the profile — not the engine code.
