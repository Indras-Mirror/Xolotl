# How Xolotl works

[Back to the README](../../README.md) · [Benchmarks](BENCHMARKS.md)

## How it fits a 552B model on a gaming PC

A MoE model only uses a few experts per token. Xolotl keeps every expert somewhere and moves the ones each token's
router asks for:

| Tier | What lives there | How it is fed |
| --- | --- | --- |
| **VRAM** | the hottest experts (sized from the VRAM actually free) | LRU cache, census-ranked, grows and shrinks per request |
| **Pinned RAM arena** | tens of GiB of experts (`cudaHostRegister`) | DMA straight to the GPU |
| **NVMe file tier** | everything else | O_DIRECT reads at the drive's ceiling (~5 GB/s), prefetched a layer ahead |

On top of that, per model:

- **DeepSeek-V4.1-Flash** - its own GSQ quantisation (every expert gets its format from calibration: ternary PTQ1_0,
  Q2_0 or Q3_K), the Engram n-gram memory read from disk, **DSpark speculative decoding** with exact speculative
  sampling and confidence-gated verify, a **bit-exact 4-bit KV cache** (`--kv-fp4`) for 512K context, and selection-only
  routing biases that keep every expert reachable. Opt-in extras: **fast mode** reads a 2-bit (Q2_K) copy of the
  biggest experts from a sidecar file (+27 % decode, short34 top-1 87.0 -> 86.2 %;
  [docs/XOLOTL_FAST.md](../XOLOTL_FAST.md)), and **Xolotl Direct** has the NVMe drive write file-tier experts straight
  into VRAM ([docs/XOLOTL_DIRECT.md](../XOLOTL_DIRECT.md)). Full write-up: **[docs/DEEPSEEK_V41.md](../DEEPSEEK_V41.md)**.
- **GLM-5.3-Flash** - KDA + MLA hybrid attention, 288 experts top-8, the abliteration LoRA applied to the dense half
  **and** the routed experts on every path (GPU, MMQ prompt chunks, CPU pool), soft pruning (every expert stays
  reachable), an elastic expert cache that lends its VRAM to long prompts. Details: [docs/glm](../glm).
- **Qwen3.8-Flash-Next** - upstream Strata 0.1.44: MTP speculative decoding, int8 KV, image input with an encoder that
  starts on the first picture and gives its VRAM back when idle. Details: [README.strata.md](../../README.strata.md).
- **Qwen3.8-27B** - native GGUF dense engine (`xolotl-dense`, built on Mirko Perrone's dense port): MMQ prompt prefill, MTP speculative decoding (draft 3),
  int8-K / int4-V cache (`k8v4`) for 200K context.

Every model is served by the same Python server with each model's own chat template and tool-call format (Qwen
XML, GLM `<arg_key>`, DeepSeek DSML), OpenAI **and** Anthropic endpoints, streaming, tool calls, conversation-prefix
reuse, a `reasoning_budget_tokens` thinking budget (32K by default), and two extras:

- **`/v1/decide`** - a one-token decision from the model's real log-probabilities (e.g. "is the user done talking?"),
  for voice turn-taking without a second model.
- **Voice sidecar** - `/v1/audio/transcriptions` and `/v1/audio/speech` proxied to an ASR / TTS server that starts on
  the first request and stops when idle, shrinking the expert cache while it runs.


## The agent loop: prefix reuse and the adaptive arena

An agent re-sends its whole conversation every step, so the server keeps the previous request's KV cache and only
prefills the new tail. On DeepSeek-V4.1 a small turn is still NVMe-bound - a +180-token turn routes to ~7,000
distinct experts, ~2,600 of which live only on the NVMe file tier. Two things cut that:

- **Small-turn prefill** (bit-identical logits): the four span-index tensors are uploaded once per prompt instead of
  per layer, and below 4,096 rows the layer prestage is filled lazily with only the routed experts. A +189-token
  turn: 9.83 -> 8.45 s.
- **The conversation-adaptive arena** (on by default; `STRATA_ARENA_REBAL=0` turns it off): routed-token counts per
  expert, decayed per prompt; after each prompt a swap plan moves the conversation's frequent file-tier experts into
  the RAM arena (evicting rarely used ones whose slot fits), and a reader thread performs the swaps in the NVMe's
  idle time - while the agent runs its tools. Routing stays exact. Measured (4K base + 8 turns, 20 s tool gaps):
  file-tier reads per small turn -21..-30 %, small-turn time 8.48 -> 7.60 s, every published slot byte-checked
  (`DS41_CHECK_ARENA=1`, 0 bad of 2,500), HumanEval/tool/agentic unchanged (21/21, 10/10, 3/3).
  Benchmark: `tools/turnbench.py <url> --gap 20`.
