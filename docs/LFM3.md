# LFM3-Omni in Xolotl

`xolotl-lfm3` serves Liquid AI's **LFM3-Omni-2B**: text, pictures, audio and video in; text and speech out. It is
Xolotl's own engine (`src/program/lfm3_main.cpp`) behind the usual server (`serve.server`, OpenAI and Anthropic
routes, tool calls, prefix reuse), built for one job: to sit beside a big model on the same card as its ears, eyes and
voice, and get out of the way when the big model needs the VRAM.

Weights: [IndrasMirror/LFM3-Omni-2B-XQ-Uncensored](https://huggingface.co/IndrasMirror/LFM3-Omni-2B-XQ-Uncensored)
(XQ-6.0 recommended, 1.29 GiB, + Liquid AI's `mmproj-LFM3-Omni-2B-Q8_0.gguf`). Any LFM3 GGUF with the same layout works.

## Measured (RTX 4090, Ryzen 7 5700X, Linux; XQ-6.0, 32K context, greedy, thinking off - LFM3 does not reason)

| what | result |
| --- | --- |
| prefill (fresh prompt, `tools/throughput.py`) | 32,796 tok/s at 5.3K · 30,959 at 7.1K · 26,049 at 14K |
| decode | ~300 tok/s (292-310 across tasks and contexts) |
| follow-up turn on a 7K conversation (+700 tokens) | 0.03 s |
| taskeval: HumanEval / tool calls / agentic | 13/21 · 9/10 · 0/3 |
| speech out (`/v1/audio/speech`) | RTF 0.08 (12x real time), first audio 0.02 s warm / 1.1 s cold |
| speech round trip (our voice transcribed by the model) | WER 0.00 / 0.00 / 0.11 (the miss: "Xolotl") |
| park / unpark | 51 ms / ~75 ms, ~1.4 GB of VRAM handed back; identical tokens either way |
| VRAM in use | ~1.2 GB engine + ~2.2 GB encoders/speech helpers while they are loaded |

Agentic 0/3 is the model, not the server: on the bug-fix loops it calls an `apply_changes` tool the agent does not have
(the server does not run undeclared tools) or writes a broken LRU cache. Tool calls themselves are 9/10.

**Parity with llama.cpp** (`tools/lfm3/parity.py`, 2,048 tokens of mixed text): with the sliding-window ring sized not
to wrap, the logits match llama.cpp's to a mean KL of 1.3e-7 with 100% the same top token. The served ring (window +
ubatch cells, so 128K of context costs the 5 sliding layers 768 cells, not 128K) differs from llama.cpp only by CUDA
flash-attention rounding, KL 1.3e-3 - a quarter of what llama.cpp itself shows between ubatch 512 and 64 (5.3e-3).
`xolotl-lfm3 --fa-selftest` measures that kernel against a float64 reference.

## How it works

- **Text model**: llama.cpp's LFM3 graph (`src/models/lfm3.cpp`, `lfm-base.cpp`) on ggml-backend CUDA: 24 layers, 13
  short-convolution layers with a carried conv state, 6 full-attention layers (K/V over the whole context), 5 sliding
  layers (window 256) in a ring. A checkpoint at the end of each prompt (conv state + rings, ~10 MB) lets a
  conversation resume after a divergent turn without re-reading it.
- **Pictures, audio, video**: `xolotl-lfm3-vision` (`tools/vision/strata_vision.cpp` built on llama.cpp's libmtmd)
  turns every image and audio chunk into rows that replace the template's `<|image|>` / `<|audio|>` / `<|video|>`
  tokens. A video is its frames plus its soundtrack (mtmd's timestamp text between frames is left out for now).
- **Speech out**: the engine's `GENA` command runs LFM3's TTS prompt; per audio frame it hands the output-normed
  hidden state to `xolotl-lfm3-speech` (`tools/lfm3/lfm3_speech.cpp`: the depthformer, MaskGIT sampling and the Liquid
  codec, ported from llama.cpp's `mtmd-helper-gen.cpp`), which returns 12 codes, the next input row and 24 kHz PCM.
  The helpers are separate processes so libmtmd's shared ggml never meets the engine's vendored ggml.
- **Speech in**: `/v1/audio/transcriptions` asks the model itself ("Perform ASR in <language>.").
- **Park**: `--park-idle S`, or `PARK` / `VRAM <mib>` from the server's `POST /v1/vram`: the weights stay in a pinned
  host copy, the state is copied to RAM and every GPU buffer is freed; the next request brings it back.
  `--peer-vram http://127.0.0.1:8099/v1/vram` asks another Xolotl server (e.g. the big model) to keep room free before
  an unpark and tells it the VRAM is back after a park.

## Beside a big model: QFN + LFM3 with Quetza's voice pipeline

A 2B omni model is most useful as the ears and voice of a larger one. `scripts/xol-qfn-voice.sh` runs both on one card
and opens QuetzaCodetl with voice mode pointed at LFM3:

```bash
scripts/xol-qfn-voice.sh              # QFN (chat) + LFM3 (ears, voice), then the CLI
scripts/xol-qfn-voice.sh --serve-only # start/attach both and exit
```

The one non-obvious step is VRAM. QFN's elastic expert cache takes all free VRAM at start (23.4 / 24.5 GB), and LFM3's
encoder then fails to load (`cudaMalloc failed: out of memory`, reproduced 2026-11-02). The script POSTs
`{"reserve_mib": 4608}` to QFN's `/v1/vram` first; QFN gives 4.4 GB back, and LFM3 loads in ~6 s (3.2 GB resident),
leaving ~1.7 GB headroom. `XOLQFN_VOICE_RESERVE_MIB` changes the amount.

Measured with both up (RTX 4090): ASR, TTS (24 kHz pcm) and `/v1/decide` all answer concurrently, and QFN keeps its
speed - a 96-token answer took 0.75 s while LFM3 streamed 3.5 s of speech. Turn-taking stays on the chat model:
`/v1/decide` is a route on that server, so it does not move with the ears.

QuetzaCodetl chooses the backends (`/voice-backend`, or `QUETZA_VOICE_BACKEND=lfm3`); the launcher sets that variable
and `QUETZA_LFM3_URL` / `QUETZA_LFM3_SERVER` / `QUETZA_LFM3_VOICE`, and `/voice-auto` starts the pipeline.

## Voice: how LFM3 speaks, and what a voice LoRA would change

LFM3 has no speaker conditioning. `male` / `female` is a **text instruction** in the TTS prompt
(`Perform TTS in American English using a male voice.`); the reference pipeline warns and ignores a speaker reference
file. Speech is generated as codec codes, not as a waveform from a speaker embedding:

1. the 2B LM emits 12 codes per 80 ms frame (12.5 Hz) autoregressively;
2. a depthformer in the mmproj fills the 12 codebooks (MaskGIT, 4 + 4 + 4);
3. the LiquidCodec decoder turns the codes into 24 kHz audio.

The pieces for a voice LoRA are in the BF16 checkpoint (`LFM3-Omni-2B/audio_tokenizer`, a `LiquidCodecEncoder` with
`encode()`): encode a target speaker's audio to codes, pair with transcripts, LoRA the LM with the codec and depthformer
frozen, merge into BF16 and requantise. That produces **one baked-in voice**, not cloning - arbitrary-speaker cloning
needs conditioning the architecture does not expose.

## Build and run

```bash
ninja -C build-xolotl xolotl-lfm3                    # the engine (needs the CUDA build: STRATA_DENSE + ggml-cuda)
tools/lfm3/build-helpers.sh ~/llama.cpp              # the two helpers, against llama.cpp f3586e6+ (built, with LFM3)
cp tools/lfm3/xol-lfm3.example.json xol-lfm3.json    # set the model, mmproj and tokenizer paths (absolute is safest)
.venv/bin/python -m serve.server --engine strata --config xol-lfm3.json --port 8132
```

The tokenizer folder is LiquidAI/LFM3-Omni-2B's `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`.

Config keys the LFM3 config uses (all opt-in; other models are unchanged without them):
`image_pad` (`<|image|>`), `vision.av` (audio and video parts), `bos_token` (`<|startoftext|>`), `thinking: false`
(a non-reasoning model: the reply is content from the first token), `hide_tokens` (LFM3's mode markers kept out of
reply text), `speech` (voice, language, max frames) and `asr_native`.

## Limits and open items

- One conversation at a time (no batched multi-slot yet); `--kv-ram` is not built.
- KV types: `--kv-type f16|q8|k8v4` is in the source; f16 is what is measured above. At 128K, F16 K/V is 1.31 GB.
- Speech sampling uses its own random stream, so waveforms do not match llama.cpp's sample for sample; a greedy
  code-for-code check against llama.cpp is still to do.
- Asked through chat ("Transcribe this audio"), the model sometimes answers in its interleaved voice mode and stops
  after the first sentence; `/v1/audio/transcriptions` returns the whole text.

## Credits

[Liquid AI](https://huggingface.co/LiquidAI) for LFM3-Omni-2B and its projector; the ggml / llama.cpp authors of LFM3
support and libmtmd (the graph, the encoders and the speech pipeline this engine follows).
