# Prune masking (`STRATA_EXPERT_KEEP`)

A task profile for the Qwen3.8 MoE engine (`strata`): a list of the experts each layer keeps, so the router only ever
picks from those. The weights on disk stay as they are; the mask only changes routing, and turning it off (unset the
variable) gives the build's normal output back. Opt-in, off by default.

It is a **hard** mask: a pruned expert can never be picked, for any token. That is the opposite of a soft profile
(a routing bias), where every expert stays reachable and only the odds move. Use a hard mask when you know the
workload and have gated the list on it (see [Building a keep list](#building-a-keep-list)); use a soft bias when you
don't.

## What it does

```bash
STRATA_EXPERT_KEEP=/path/to/keep.txt build-xolotl/strata --serve ...
```

The file lists, per MoE layer, the experts that **stay**:

```
# keep 384 of 512, office profile
0: 0 1 3 4 5 7 8 9 10 12 15 ...
1: 0 1 3 4 5 6 10 11 12 14 ...
...
47: ...
```

- One line per layer: `<layer>: <expert ids>`, ids separated by spaces or commas. `#` starts a comment.
- A layer with no line keeps every expert, so you can mask only some layers.
- Each masked layer must keep at least 10 experts (the model routes 10 per token). A bad layer or expert number stops
  the engine at start with a message saying which one.

Before the router's top-k, the logits of pruned experts are set to -1e30. The softmax, the top-10 pick and the
weight renormalisation then see only the kept experts: the same routing you would get from a GGUF with those router
rows deleted, but on the original file. This is applied everywhere the router runs: decode, the speculative verify
pass and prefill. The start-up log confirms it:

```
[strata] STRATA_EXPERT_KEEP .../keep-320.txt: 48 of 48 layers masked, 320.0 of 512 experts kept per masked layer
strata generate: STRATA_EXPERT_KEEP: 9216 pruned pairs ranked last in the profile
```

## What the engine does with it

Masking the router alone would already change the output. The speed comes from the engine also knowing which experts
can never be asked for:

| Where | With a mask |
| --- | --- |
| VRAM residency profile (`--expert-profile`) | pruned experts are ranked last, so the VRAM cache fills with experts that can actually be routed |
| `--resident-experts` (low-RAM mode) | pruned experts are left out of the page-locked RAM copy: 24.79 -> 11.17 GiB on QFN45 at keep-320 |
| Prefill stream walk | pruned experts are not copied over PCIe (they were ~40 % of each prompt chunk's traffic at keep-320) |
| Host prefill plan | counts kept experts only, so it no longer plans a file tier for experts that are never read |

Compute per token does not drop: every token still runs 10 experts. The gains are fewer cache misses and less data
moved.

## Building a keep list

The method, in four steps:

1. **Pick a workload that represents the task.** Ours: a set of real spreadsheet-agent conversations (tool calls,
   sheet reads, long system prompts), split by date into a profile set and a later held-out test set.
2. **Count routing over whole conversations.** Run each conversation through the full, unmasked model teacher-forced
   (feed the real tokens, don't sample) and count how often every (layer, expert) pair is picked, at **every
   position**. Decode-only routing traces miss the prompt, and in agent work the prompt is most of the tokens.
3. **Keep the top N per layer** by count.
4. **Gate it on held-out data** before trusting it. Run the test set through the masked model and the full model and
   compare their next-token distributions on the reply tokens: mean KL divergence and top-1 agreement. Pick the
   smallest N that stays close.

QFN45 (512 experts per layer, 48 layers), office profile, held-out gate:

| Keep per layer | Routed mass kept (profile set) | KL vs full model | Top-1 agreement | Verdict |
| --- | --- | --- | --- | --- |
| 384 of 512 | 98.9 % (worst layer 96.3 %) | 0.057 | 94.9 % | current choice |
| 320 of 512 | 96.5 % (worst layer 91.2 %) | 0.080 | 93.9 % | passed the gate, but see the task runs |
| 256 of 512 | - | 0.159 | 90.8 % | too far |

The gate set is small (two held-out exchanges, 1,338 reply tokens), so read it as a direction, not a precise
number. keep-384 also cuts the extra loss per token (dNLL +0.031 -> +0.007) and the "unsure" tokens (17 -> 9 per
1,000).

A keep-all list (every expert listed) matches the unmasked output within normal run-to-run variation, which is a
quick way to check the plumbing before testing a real list.

## Results (QFN45, RTX 4090, Ryzen 7 5700X, 110 GB RAM)

QFN45 is our 45 GB Qwen3.8 MoE quant. Throughput on a 13.8K-token prompt (`tools/throughput.py`, two runs each,
20 Nov build):

| Mode | Prefill tok/s | Decode tok/s |
| --- | --- | --- |
| Normal, keep-320 | 3,704 / 3,896 | 197 / 214 |
| Normal, no mask | 3,404 / 3,512 | 156 / 163 |
| Low-RAM (`--resident-experts`), keep-320 | 3,702 / 3,830 | 147 / 158 |
| Low-RAM, no mask | 2,590 / 2,786 | 106 / 148 |

- **Normal mode:** decode +26-31 %, from a better VRAM cache hit rate (95-97 % with the mask). Prefill +9-11 %.
- **Low-RAM mode:** the mask halves the experts held in RAM (peak RSS 68.8 -> 41.1 GB, time to first token
  22.7 -> 11.9 s) and brings prefill up to normal-mode speed.

**On real tasks.** Two multi-step spreadsheet jobs (read a cable cutsheet, find and fix planted errors, build label
sheets), each scored against a fixed set of checks:

| Job B runs | Checks | Job time |
| --- | --- | --- |
| No mask, last three runs | 12/12, 12/12, 12/12 | 480 s (last run) |
| keep-320, low-RAM mode | 8/12 | 433 s |
| keep-320, normal mode | 12/12, 11/12 | 382 s, 404 s |
| keep-384, normal mode | 12/12 | 500 s |

- Job A: 12/12 with keep-320.
- On job B the 8/12 was one wrong decision that did not come back in later runs. One specific false alarm
  (reporting a problem that isn't there) showed up in 2 of the 3 keep-320 runs and in none of 5 unmasked runs. The
  keep-384 run did not raise it. That is one run, so it is a lead, not proof; keep-384 keeps more of the routing
  (98.9 % vs 96.5 %) for a smaller speed gain, which is the safer side.
- Job time follows how many steps the model takes more than engine speed: the keep-384 run made 88 requests and
  wrote ~50K tokens, the keep-320 runs ~61 requests and ~41K tokens each, at the same ~152 tok/s decode.

## Caveats

- **Task-specific.** A list built from office work keeps the experts office work uses. On coding or other languages
  it may route worse than the full model. Build and gate a list per task.
- **Gate first.** KL and top-1 on held-out data, then real tasks; a list that looks fine on throughput can still
  drop the odd decision (see above).
- **Hard, not soft.** Pruned experts are unreachable. If you need every expert reachable, use a routing bias
  instead.
- **Qwen3.8 MoE engine only** for now (`strata`). The DeepSeek-V4.1 engine has rank files and routing biases (soft
  profiles) but no hard mask.
