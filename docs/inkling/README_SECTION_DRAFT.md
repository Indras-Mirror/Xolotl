# DRAFT: Inkling-2-Mini section for the Xolotl README

Two pieces: one bullet for the "Latest" list, and one short section. Both are drafts; the section stays out of
the README until the engine passes S3 and the numbers exist.

---

## Latest bullet (for the top list)

- **DATE TBD** - [Inkling-2-Mini](docs/INKLING.md) (Thinking Machines, 26B total, about 3B active) on its own
  engine: a looped MoE that runs layers 6 to 17 three times from 24 layers of weights, 128 experts at top-8, no
  RoPE, and a 1M-token context, through the usual server. The graph is ported from llama.cpp PR #32694 by Pedro
  Cuenca (MIT). Speed and quality numbers TBD.

---

## Section

## Inkling-2-Mini

Thinking Machines' Inkling 2 Mini is a 26B-parameter looped MoE with about 3B active parameters: 24 physical
layers where the middle 12 run three passes, 128 routed experts at top-8 with a sigmoid router, a learned
relative-position bias instead of RoPE, and a 1M-token context. `xolotl-inkling` runs it natively, from any
Inkling-2-Mini GGUF, and it is the next abliteration and XQ target.

| | |
|---|---|
| parameters | 26B total, about 3B active |
| layers | 24 physical, 48 applications per token (layers 6 to 17 run 3 times) |
| experts | 128 routed, top-8, no shared expert, SiTU |
| context | 1M |
| decode / prefill, RTX 4090 | TBD |
| Gauntlet v3-small | TBD |

The forward graph is ported from llama.cpp [PR #32694](https://github.com/ggml-org/llama.cpp/pull/32694) by
Pedro Cuenca (MIT), on top of the earlier Inkling 2 work in llama.cpp and ggml. Xolotl adds its own engine
around it: the serve line protocol, prompt reuse, the host-side relative-position inputs and masks, and decode
state for the 48 cache slots. One bug in the loop path is fixed here: passes 2 and 3 now apply the per-expert
NVFP4 scales when the GGUF keeps native FP4 experts, which only happens on files that skip the upstream
dequantising converter.

Adapters, the XQ requantisation and the abliterated build are measured against the parent, the official
checkpoint converted exactly and run on this engine, on the same box and harness. Numbers go here and in
`docs/xolotl/BENCHMARKS.md` when they exist.
