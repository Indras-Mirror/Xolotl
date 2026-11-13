# Muse Glimmer in Xolotl

`xolotl-glimmer` serves Meta's **Muse Glimmer 30B** (Meta Superintelligence Labs, Apache 2.0): a dense 30B model built
for local agent and tool-use work, with a picture encoder. It is Xolotl's own engine (`src/program/glimmer_main.cpp`)
behind the usual server (`serve.server`: OpenAI and Anthropic routes, tool calls, prefix reuse), at the full **512K
context** on one 24 GB card.

Weights: any Muse Glimmer GGUF (llama.cpp arch `muse-glimmer`) and its `mmproj` for pictures. Measured here with
darkc0de's abliterated `Muse-Glimmer-30B-heretic` as bartowski's IQ4_NL GGUF (16.2 GB) + the f16 mmproj (3.9 GB).

## Measured (RTX 4090, Ryzen 7 5700X, Linux; IQ4_NL, 512K context, q8 K/V)

| what | result |
| --- | --- |
| VRAM | 14.4 GiB weights + 3.5 GiB K/V at 512K (the 0.7 GiB token table stays in RAM, as llama.cpp keeps it) |
| served chat / tool call / tool round trip / thinking off / picture | all correct (`get_weather{"city":"Melbourne"}`; "a red circle, a blue square, and the number 42") |
| prefill (6.5K / 26K-token prompt) | 3,133 / 3,110 tok/s |
| decode (after 6.5K / 26K tokens) | 47.8 / 46.3 tok/s |
| decode with `--ngram 8`, output that copies its input (code edit / quoting a document) | 116-216 / 275-279 tok/s (2.4-5.7x) |
| decode with `--ngram 8`, free writing (a story) | 48.4 tok/s (0.98x: little to copy) |
| decode with `--dflash` (Meta's drafter, 15 drafts a pass): quoting a document / a code edit | 341-379 / 186-254 tok/s (7.3-8.0x / 3.9-5.3x) |
| decode with `--dflash`, 7 drafts a pass: an explanation with code / a story | 85-113 / 62-65 tok/s (1.8-2.4x / 1.3-1.4x) |
| decode with `--dflash` adaptive (the default: about 3 past the recent accepted count, doubled after a fully kept block): quoting / code edit / story / explanation | 358-382 / 177-214 / 64-69 / 87-90 tok/s |
| ElixirOffice in-app bench (14 spreadsheet tasks, ngram + DFlash) | 13/14 in 291 s (fails only the combined clean-up task) |
| next turn on a reused prefix (+191 / +159 tokens) | 0.1 s |
| card memory while serving (whole GPU, nvidia-smi) | 19.2 GB |
| parity with llama.cpp, 4,096 tokens | mean KL 1.0e-3, 99.0 % same top token at equal batch size; as close as llama.cpp is to itself (see below) |
| for comparison: llama.cpp wrapper, same GGUF (ubatch 32, TBQ4 K/V) | about 763 tok/s prefill, 39-42 tok/s decode |

**Parity** (`tools/glimmer/parity.py` against `glimmer_ref_dump`, llama.cpp with Muse Glimmer support, same GGUF,
every 8th of 4,096 positions). Batch size alone changes llama.cpp's own output by about as much as the two engines differ,
so the gap is GPU rounding, not a bug (same story as [LFM3](LFM3.md)):

| comparison | mean KL | same top token |
| --- | --- | --- |
| llama.cpp ubatch 512 vs llama.cpp ubatch 256 (its own noise) | 1.50e-3 | 97.9 % |
| Xolotl vs llama.cpp, both ubatch 256 | 1.00e-3 | 99.0 % |
| Xolotl vs llama.cpp, ubatch 512 | 1.44e-3 | 98.1 % |
| Xolotl with the sliding ring never wrapping vs llama.cpp ubatch 512 | 1.36e-3 | 98.6 % |

Both engines drift a little more once the sliding window fills and wraps (llama.cpp against itself: 0.76e-3 inside the
first window, 2.7e-3 after it wraps), so the ring is not the cause either.

## How it works

- **The model**: 52 dense layers (n_embd 6656, 32 query heads sharing 2 K/V heads of 128, SwiGLU 19968). Three layers in
  four are sliding-window layers (window 2048, RoPE base 500000); every fourth is a **global layer with no position
  encoding**. The embeddings get an unweighted RMS norm, attention has QK-norm and a sigmoid gate on its output, the
  post norms use eps 1e-8, and the logits are scaled by 0.196 and soft-capped at 20. The graph is llama.cpp's
  (`src/models/muse-glimmer.cpp`) on ggml-backend CUDA.
- **512K context**: the 13 global layers hold the whole context's K/V (q8: about 6.9 KB a token, 3.5 GiB at 512K); the 39
  sliding layers hold a ring of window + ubatch cells (2,560). Long context needs no RoPE scaling: the global layers have
  no positions, and llama.cpp gives the sliding layers their trained scale whatever `--rope-scaling` says, so a YaRN
  setting changes nothing for this model.
- **Prefix reuse**: a request that extends the last one reads only the new tokens; a divergent turn resumes from the
  checkpoint taken one token before the end of the previous prompt (the sliding rings, ~100 MB).
- **Turn format**: Glimmer answers in messages to channels - `to=self` (its reasoning), `to=user` (the answer),
  `to=<tool>` (a tool call in ATEM markup, `<atem:invoke name="...">`). `serve/atem.py` reads them as the reasoning,
  answer and tool calls the rest of the server knows, as it streams. `reasoning_effort` becomes the template's
  "Reasoning strength"; thinking off starts the reply at the answer.
- **Drafting** (`--ngram K`, on in the example config at 8): the engine looks up the last 4, 3 or 2 tokens earlier in
  the context and proposes up to K tokens that followed them, then checks them in one pass. Each row is sampled with the
  same counter plain decode would use, and a draft is kept only when it equals that sample, so the reply is the tokens
  plain decode would pick (up to batch-size rounding: replayed through a fresh pass, no emitted token sits more than 0.4
  logit below the top pick, same as with drafting off). It pays off where the reply repeats the context: code edits,
  quoting, tool calls with long arguments. A request can lower it with `ngram=K`.
- **DFlash** (`--dflash <dflash.gguf>`): Meta ships Glimmer with a block drafter (on in the example config; `dflash-Muse-Glimmer-30B-Q4_K_M.gguf`,
  1.6 GB, in meta-models/Muse-Glimmer-30B-GGUF; five sliding-window layers of the model's width). Every forward pass of
  the model taps the residual stream entering layers 2, 14, 26, 38 and 50, fuses the five rows and writes the drafter's
  K/V for those positions into its own 2,560-cell ring, inside the same GPU graph. A draft is one non-causal pass of the
  drafter over the last token plus 15 mask tokens, read through the model's own token table and output matrix; the
  model then checks the block as above (kept only where it equals the sample). 1.5 GB of VRAM + a 50 MB ring; at 512K
  q8 that leaves 2.8 GB free. A request can set `dflash=K` (0 = off, then n-gram drafting is used).
- **Sampling**: temperature 0.6, top_p 0.9, top_k 20 by default (`arch: muse-glimmer`). Above 0.6 the model stops writing
  the tool-call markup.
- **Pictures**: `xolotl-glimmer-vision` (`tools/vision/strata_vision.cpp` on llama.cpp's libmtmd) turns a picture into
  rows that go between `<|image_start|>` and `<|image_end|>`, as mtmd lays it out. At 512K the encoder runs on the CPU
  (about 13 s for a 640x480 picture): beside the engine its 4.3 GB on the GPU leaves no room for a long prompt.
- **Park**: `--park-idle S`, or `PARK` / `VRAM <mib>` from the server's `POST /v1/vram`, frees every GPU buffer (weights
  kept in pinned RAM); the next request brings it back.

## Run it

```bash
tools/glimmer/build-helpers.sh ~/llama.cpp build      # the picture encoder + the parity reference, from a llama.cpp
                                                      # checkout with Muse Glimmer support
python tools/strata_tokenizer.py --gguf <glimmer.gguf> --out <dir>    # the tokenizer (pre-tokenizer llama4)
python -m serve.server --engine strata --config tools/glimmer/xol-glimmer.example.json --port 8133
```

In the tokenizer folder's `chat_template.jinja`, write a picture as `<|image_start|><|patch|><|image_end|>` (the GGUF's
template writes `<|patch|>` alone); the config's `image_pad` and `vision_start` name those two tokens.

## Credits

Muse Glimmer by Meta Superintelligence Labs. The engine follows llama.cpp's Muse Glimmer support (graph, chat parser,
mtmd projector; ggml-org/llama.cpp #26841). The abliterated build measured here is darkc0de's, quantised by bartowski.
