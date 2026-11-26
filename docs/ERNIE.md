# ERNIE-5.5-Flash on Xolotl (preview)

> **Draft / preview.** The engine runs ERNIE-5.5-Flash correctly and serves it through Xolotl's OpenAI / Anthropic
> server, but it has been tested on one GGUF layout so far and the speed work is still going on. Faster decode
> (speculative decoding, the 2-bit expert shadow) and a quant of our own come next. Numbers and flags may change.

Baidu's [ERNIE-5.5-Flash-Preview](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) (Apache 2.0) is a 280B MoE
with about 14B active parameters per token. `ernie_generate` is Xolotl's engine for it. It reuses the DeepSeek-V4.1
routed-expert tier (VRAM cache, RAM arena, NVMe file tier) with ERNIE's own geometry and activation, and runs the
dense half as per-layer ggml graphs.

| | |
|---|---|
| layers | 43: layer 0 dense, layers 1-42 MoE |
| attention | 37 HCA layers (compressed KV, 128-token windows) + 6 MLA/DSA layers (7, 14, 21, 28, 35, 42) |
| experts | 512 routed, top-10, plus 1 shared. The experts work in a 2048-wide latent (`fc1_latent_proj` 4096 -> 2048, then `fc2_latent_proj` back), SiTU activation |
| residual | 4-stream mHC (hyper-connections) |
| expert weights read per token | about 5.3B |

## Get the model

Our quant: **[IndrasMirror/ERNIE-5.5-Flash-XQ-Uncensored](https://huggingface.co/IndrasMirror/ERNIE-5.5-Flash-XQ-Uncensored)**
(99.9 GB; routed experts at 2.65 bits per weight on average in IQ2 / IQ3 types, everything else Q8_0; uncensored). It
fits a 24 GB GPU plus ~96 GB of RAM with every expert resident. It comes as 10 parts; join them after the download:

```bash
hf download IndrasMirror/ERNIE-5.5-Flash-XQ-Uncensored --local-dir ernie-xq
cd ernie-xq && sha256sum -c SHA256SUMS
cat ERNIE-5.5-Flash-XQ-Uncensored.gguf.part-* > ERNIE-5.5-Flash-XQ-Uncensored.gguf && rm ERNIE-5.5-Flash-XQ-Uncensored.gguf.part-*
```

## Which GGUFs it reads

The tensor names and metadata of **llama.cpp PR [#31925](https://github.com/ggml-org/llama.cpp/pull/31925)**
("model : add ERNIE 5.5", by FakerVSdonk), at its commit `7d9d5c351`. That PR is still a draft, so its names can
change before it merges; if a GGUF made by a later version of the PR fails to load, please open an issue with the
error line.

Tested so far:

- MXFP4 routed experts with a Q8_0 dense half (154 GB), converted from Baidu's FP8 safetensors with
  `tools/ernie/ernie_gguf.py`.
- [our XQ file](https://huggingface.co/IndrasMirror/ERNIE-5.5-Flash-XQ-Uncensored) (99.9 GB, IQ2 / IQ3 routed experts).
- The engine's expert kernels cover the usual ggml types (Q2_K-Q8_0, IQ1_M-IQ4_XS, MXFP4) through the same tier
  DeepSeek-V4.1 uses, but on ERNIE only MXFP4 and the IQ2 / IQ3 types of our XQ file have been run end to end. Other quants should work; reports welcome.

## Correctness

Checked against the PR's own llama.cpp build on the same file:

- next-token KL 0.004 on a near-tie token over a full 43-layer pass
- teacher-forced NLL 0.093 vs llama-perplexity's 0.080 on the same rows (the gap is ggml's Q8 activation rounding,
  traced tensor by tensor with `tools/ernie/ernie_compare.py`)
- tokenizer: identical ids to `llama-tokenize` on the test strings (`tools/v41/engine/test_ernie_tokenize.sh`)

`tools/ernie/README_REF.md` has the numpy reference (`ernie_ref.py`, straight from the FP8 safetensors) and the
tensor-by-tensor comparison against the PR.

## Speed (RTX 4090, Ryzen 7 5700X, 110 GB RAM, one NVMe)

| | 154 GB MXFP4 file | our [99.9 GB XQ file](https://huggingface.co/IndrasMirror/ERNIE-5.5-Flash-XQ-Uncensored) |
|---|---:|---:|
| decode | 4.46 | 13.4-14.4 (served 14-15) |
| prefill | 12.9 | 94 (1.9K prompt, 128-token chunks); served 115-120 (6.5K-25K prompts, 256-token chunks) |

The MXFP4 file is bigger than RAM plus VRAM, so about 15 % of its expert reads go to the NVMe drive. The XQ file fits
in RAM plus VRAM (80 GiB arena). Per token ERNIE routes 10 experts in each of 42 layers; about 45 % of those are computed
on the CPU, and that is now the limit. Next: DSpark speculative decoding (in progress) and GPU expert compute for prompt
chunks.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target ernie_generate -j

# the best decode settings measured so far
./build/ernie_generate -m ernie-xq/ERNIE-5.5-Flash-XQ-Uncensored.gguf -p "Write a haiku about rivers." -n 128 \
    --experts gpu --cuda --slots 2300 --threads 7 --arena-gib 80 --ds41-decode 0 --prefill-chunk 256
```

| flag | what it does |
|---|---|
| `--experts gpu` / `--cuda` | routed experts on the GPU tier / the dense graphs on the GPU |
| `--arena-gib G` | RAM for routed experts (set it to what you can spare) |
| `--slots N` | VRAM expert slots (2300 leaves room for 256-token prompt chunks on a 24 GB card); 0 = all free VRAM minus 1 GiB |
| `--prefill-chunk 256` | prompt passes of at most 256 tokens (512 still runs out of memory at ~25K context) |
| `--count-routes F` / `--rank-counts F` | record which experts a workload uses, then fill VRAM and RAM with the most used first. Without a profile the cache fills in index order and decode is about 3x slower |
| `--ds41-decode 0` | one-token steps through the DeepSeek-V4.1 decode path, with the next layer's experts guessed ahead |
| `--batch F`, `--batch-out F`, `--batch-logits F` | many prompts in one load (0x1E-separated), greedy continuations and last-row logits |
| `--abl F`, `--abl-scale S` | directional ablation at load from a direction file (format in `ernie_graph.hpp`) |
| `--kv-mla f16\|q8_0\|q4_0` | the MLA layers' KV rows (~96 % of the KV at long context): q8_0 halves it, q4_0 quarters it (Hadamard-rotated by default, `--kv-hadamard 0\|1`) |
| `--vram-lru` / `--arena-adapt` | the VRAM cache / the RAM arena follow the conversation (least-recently-used replacement) |
| `--arena-skip-resident` | experts the VRAM cache holds stay out of the RAM arena (room for the next-ranked ones) |
| `--skip-miss W` / `--skip-file W` | decode: drop a VRAM-miss (or NVMe-only) expert weighing under W of the token's routing weight. A quality trade: measure first |
| `--remote-device N` | a second CUDA card holds and computes the next-ranked experts |
| `--stats` | the tier's hit / miss / timing counters |

Rendered chat text uses ERNIE's own special tokens; pass `--no-bos` when the text already starts with them.

## Serving

```bash
python3 tools/strata_tokenizer.py --gguf ERNIE-5.5-Flash.gguf --out ernie-tokenizer      # the tokenizer pack (pre ernie5)
# edit configs/xol-ernie.json: the GGUF path, the tokenizer folder (ernie-tokenizer/tokenizer), --arena-gib
python3 -m serve.server --engine strata --config configs/xol-ernie.json --port 8140
```

`ernie_generate --serve` speaks the server's engine protocol: prompt progress, STOP, sampling (temperature, top_p,
top_k up to 64, min_p, seed), repetition / frequency / presence penalties and DRY (`dry_multiplier`, `dry_base`,
`dry_allowed_length`, `dry_penalty_last_n`, llama.cpp's names, per request or in the config's `sampling`).
A follow-up turn reads only its new tokens: the engine keeps the conversation and a snapshot at the last prompt's end.
The prompt is Baidu's own renderer (`serve/ernie_encoding.py`, vendored from the model repo), with
`reasoning_effort` low / medium / high and thinking on or off; `serve/ernie.py` turns ERNIE's
`<ernie:think>` / `<ernie:tool_call>` output into the server's reasoning, answer and tool-call events.
`--count-routes` adds every request's routing to the profile `--rank-counts` seeds from, so the caches learn what
you use. One request at a time for now.

## Credits

Model by Baidu. GGUF layout and the reference implementation we checked against: llama.cpp PR #31925 by
FakerVSdonk. Engine work in this repo builds on the DeepSeek-V4.1 engine and its contributors (see
[CREDITS.md](../CREDITS.md)).
