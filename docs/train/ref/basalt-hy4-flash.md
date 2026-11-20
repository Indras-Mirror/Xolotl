# Hy4-flash

Tencent's Hy4-flash (`tencent/Hy4-flash`). Runs with the same command as any model: `basalt serve PACK`. What it supports next
to the others: [models](index.md). Rounding and quality: [precision](../precision/hy4-flash.md).

## Pack

One file, `Hy4-flash-NVFP4.basalt` (158.7 GB), built from Tencent's BF16 checkpoint (Tencent Hy License 1.0) with the vision
projector and the DFly2 drafter:

- Routed experts: NVFP4, rounded once from BF16 (83.6 GB).
- Attention, latent, shared-expert, dense-FFN, Engram and indexer projections and the token embedding: Q8_0 of the BF16 values.
- Output head: BF16 in the file; the engine rounds it to Q8_0 as it loads. Norms, hyper-connections, router and biases: F32,
  as stored.
- Engram table: E4M3 codes, one byte per value, rounded once from BF16 (63.4 GB). It stays in the file and is read by row.
- Vision projector (BF16, 1.0 GB) and the drafter (BF16 in the file, 1.7 GB), whole.

Download it from [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt)
(`hf download jesdga/Hy4-flash-Basalt Hy4-flash-NVFP4.basalt --local-dir models`), or build the same file yourself:
[model file](../model-file.md#hy4-flash-nvfp4-pack). A test pack from AngelSlim's Q4_K_M GGUF converts too
([model file](../model-file.md#hy4-flash-test-pack)).

## Architecture

The `hy4flash` architecture (code in `src/models/hy4flash`): 42 layers over four identity hyper-connection streams.

- **Attention:** MLA over one shared latent. A position's cache row is 576 values (the normed 512-value latent and 64 roped
  values); attention has a learnable sink and a sigmoid output gate. 12 sparse layers (0, 1, 5, 9, ..., 41; 32 heads) keep
  every position's row, and an indexer picks the 2,048 positions each row attends: it pools one key per 4 positions and keeps
  the best 511 groups plus the current one. Up to 2,048 positions it keeps them all. 30 sliding-window layers (64 heads) attend
  the last 512 positions.
- **MoE:** layer 0 is a dense FFN; layers 1 to 41 have 512 routed experts (top 16) and one shared expert. The routed experts
  run in a 1,536-wide latent: the layer projects its input down to it and the experts' sum back up, so an expert's three
  matrices are 1,536 x 1,536.
- **Engram:** layers 1, 11, 21 and 31 add rows of a hashed n-gram table, 16 rows of 96 values per token and layer.

## Memory

- **Experts:** the shared expert tier, placed as DeepSeek's ([memory](deepseek41.md#memory)): the main card holds copies of
  the most routed experts, each second card holds experts of its own, and system RAM holds the rest (the routed experts are
  77.8 GiB), computed there by the CPU. The start line `expert homes of 512 per layer: ...` gives the split. The start places
  experts by the shipped routing profile `data/profiles/hy4flash-42x512.prof`, so the first request after a start finds the
  most routed ones resident. The early layers route more widely and get more of the main card's copies (layers 0 to 3 about
  1.4 times the mean, the quietest 0.73), the same total.
- **Cards:** the main card holds the dense weights (6.76 GiB with the Q8_0 head), the caches and the drafter (0.87 GiB). Up to three more cards
  (`--gpu 0,1,2,3`) each hold the experts of one block of layers; later cards idle.
- **Engram table:** read by row from the pack, so it takes no RAM of its own; the OS file cache keeps the hot rows.
- **Context:** `--max-context` from 1,024 up to 1,048,576, default 8,192. The 12 sparse layers keep a row of every position;
  the window layers' rings do not grow.
- **Cache precision:** `--kv fp16` (default) keeps the rows in fp16; `--kv fp8` stores vLLM's 656-byte FP8 row
  ([precision](../precision/hy4-flash.md#attention-caches)). With the pooled indexer keys, a position takes:

  | | Per position | 128K | 256K | 1M |
  | --- | ---: | ---: | ---: | ---: |
  | `--kv fp16` | 14.6 KB | 1.8 GiB | 3.6 GiB | 14.3 GiB |
  | `--kv fp8` | 8.6 KB | 1.1 GiB | 2.1 GiB | 8.4 GiB |

- **Elastic caches:** as Flash-Next's. With `--max-context` above 32,768 a session backs its first 32,768 positions at start;
  the start is refused only when those need more than half the memory the weights leave. A prompt or a generation that needs
  more positions takes them from the expert tier (its spare slots first, then the slots it lends), and a later prompt that needs
  fewer gives them back. A prompt whose positions would leave the tier below a prompt chunk's needs is refused with an error and
  the engine serves the next request. The start line `elastic KV: ...` and `/v1/status` (`context.elastic_kv_ceiling`) give the
  positions that fit.
- **VRAM:** `--vram-reserve-mib` and `POST /v1/vram` set what stays free, as for DeepSeek.

## Long prompts

- Chunks are up to 16,384 tokens, halved to fit memory. Inside a chunk the attention runs in steps of 2,048 rows.
- A prompt of 1,024 tokens or more borrows only the tier slots its buffers and the expert ring need; the ring copies the next
  layer's experts to the card while a layer computes. The slots it leaves keep their experts, and the next turn finds them. A
  shorter prompt keeps the card's slots and puts its buffers beside them; the CPU takes RAM experts of 8 rows or fewer.
- With a second card, each chunk's experts split between the cards: the second card computes its own and the RAM experts its
  link prices lower.
- The indexer scores every pooled key for every row, so its cost grows with the context. At 1,015,207 tokens with `--kv fp8`
  the prompt read in 486 s (2,090 tok/s, both cards at their default power limits) and a needle at 50 % was found. Needles at
  10, 50 and 90 % of a 253,410-token prompt were found with either cache format.

## Drafting

On when the pack has the DFly2 drafter. One block pass drafts up to 5 tokens and states each one's chance of acceptance, and the
draft policy picks the window by expected tokens per millisecond, as for DeepSeek
(`include/basalt/spec/draft_policy.hpp`). `--spec N` caps the window (default 6: the last accepted token plus the whole block);
`--spec 0` turns it off.

The drafter is five dense MLA layers that read target layers 25, 29, 33, 37 and 41: each attends the positions that target layer's
indexer selected, in that layer's cache, so its cost does not grow with the context. It keeps no cache of its own, and
the target checks every draft: output is plain decoding's ([precision](../precision/hy4-flash.md#drafting)). Its projections
are rounded to Q8_0 as they load, and its drafts read the target's own Q8_0 head. It works with either `--kv`. The start line:

```
basalt hy4flash: drafter: 5 layers reading target layers 25..41, 0.87 GiB on the GPU (Q8_0 projections)
```

## Parallel slots

Not measured. `--parallel N` starts N sessions, each with its own caches, rings and drafter scratch, over the one expert tier;
their windows run one after another, with no shared pass through the weights.

## Images

`--vision dynamic` only ([vision](../running.md#vision---vision)). The encoder (a SigLIP2-derived ViT and its projector, 928 MiB)
is staged once in pinned RAM; a request's pictures borrow room on the main card from the expert tier while they are encoded,
and the tier takes it back. A picture is resized (Pillow's bicubic, as transformers' processor) to 4 to 4,096 merged tokens; one
more than 200 times longer than wide is refused. Its rows take three-axis rotary positions, and the Engram rows of picture
positions are masked.

## Chat, thinking and tools

- **Template:** the pack's own Jinja template. Its special tokens carry a suffix (`:opensource` in Tencent's release, another in
  AngelSlim's), which the server reads from the template, so the stop token, thinking tags, tool tags and image tokens follow the
  pack. Token text inside a message stays text.
- **Thinking:** `reasoning_effort` `none` renders `no_think` (an empty `<think></think>`); `low`, `medium` and `high` render
  `high`; `max` renders `max`; a number is refused. `chat_template_kwargs.preserve_reasoning` is the template's
  `preserved_thinking`. A thinking budget closes the thinking with `</think>`.
- **Tools:** the model writes
  `<tool_calls><tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call></tool_calls>`. Each value is read by the
  request's schema for that parameter (text where a string is allowed, JSON otherwise); several calls per turn are supported.
- **Sampling:** the pack's defaults are temperature 0.9, top_p 1.0 and no top_k; a request's fields win.

## Lossy flags

Off by default. `--route-prior` and `--route-tail-skip` change output, and each is named in one start warning. What they do on
every MoE model: [quality](../precision/quality.md#lossy-routing-flags); what they cost and gain here:
[precision](../precision/hy4-flash.md#lossy-routing-flags). Without them, output still depends on which experts are resident, so a
different tier size can change the text ([precision](../precision/hy4-flash.md#experts-and-placement)).

## Limits

- No video.
- No layer split: further cards hold experts only.
- After a very long prompt the main card holds fewer expert slots, as the caches took their room: 75 of 129 per layer after
  1,015,207 tokens with `--kv fp8`, against 114 to 122 after 253,410. Decode then computes more experts on the CPU.

## Speed

Serve `bench.sh` (the second pass of a server), RTX 5090 at 400 W with an RTX 5060 Ti at 180 W, the published pack:

| | Result |
| --- | --- |
| Decode: conversation, struct, prose | 92.2, 171.5, 83.8 tok/s |
| Prompt read | 3,454 tok/s |
