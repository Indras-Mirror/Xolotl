# Several requests at once (batch slots)

By default Strata serves **one request at a time**: the others wait in the server's queue. With `"parallel": N`
(the engine's `--batch N`, also spelled `--slots N`) the engine keeps up to N conversations open and decodes them
**together**: every verify window then carries one token of each conversation, so the dense weights, the shared
expert, the head and every routed expert two conversations share are read once per window for all of them.
Combined with a layer split across several GPUs and `--batch-groups`, the cards also work on different
conversations at the same time instead of waiting for each other.

It is opt-in and changes nothing when the options are absent (#465; the engine part is PR #559).

## Turning it on

One GPU: add `"parallel": 2` to the model's config (`strata-<model>.json`) and restart, or run setup with
`--parallel 2`. Setup recommends it only where it does not cost speed (below); any number you ask for is kept as
asked, with a note when it is more than setup would recommend.

```
"parallel": 2
```

With MTP (`--mtp` and `--spec`), `--batch-mtp` (in the config's `args`, or `STRATA_BATCH_MTP=1` in the server's
environment) lets each batch slot verify one MTP proposal per window. It is opt-in; without it the batch behaviour
described below is exactly the one without MTP. It needs VRAM per slot for the draft state and buffers, so check the
engine's free-memory log before using it on a smaller card. If it cannot run (one slot, no `--mtp`, a layer split with
`--batch-groups` above 1 or with helper-GPU expert caches or `--remote-expert-opt`, a layer split with two stages on one
GPU) the engine says so and batches as usual. RTX PRO 5000 owners measured +31% to +39% total throughput with 2 to 4
clients on one GPU (a RX R9700 run too).

**With a layer split** (e.g. `"layer_split": "24"` on two GPUs, `"parallel": 2`, `--spec 4 --mtp ...`) it works the
same way, with the slot drafters on the **last stage's GPU** (where the solo drafter and the head are): each slot gets a
drafter that shares the solo drafter's weights and owns only its K/V ring and buffers (the engine logs `--batch-mtp: N
slot drafters on CUDAk (X MiB ... each)` at start; the K/V ring of each also keeps a pinned-RAM copy, as the solo
drafter's does). A window carries two rows per active slot (its token and its proposal) through every stage, so at most
four slots are in one window; more slots (`--batch` up to what fits) rotate through. The windows run through the stages
one after the other (`--batch-groups 1`): on 0.1.41 a layer split with `--batch` 2 or more pipelines the slots in groups
by default, and the pipelined path does not run the slots' MTP drafts, so with `--batch-mtp` and no `--batch-groups` the
engine (0.1.43.1) pipelines and leaves `--batch-mtp` off, and says so (0.1.42/0.1.43 ran one serial group with it: 4 streams on 2x A4000 aggregated 63.9 tok/s against 91.8 pipelined, 85.8 pipelined without `--batch-mtp`; #2038). `--batch-groups 1` or `STRATA_BATCH_MTP_SERIAL=1` keeps `--batch-mtp` with serial windows, which is the better choice for about two streams as measured below; `--batch-groups G` (G above 1) or `--batch-groups auto` given on the command line
keeps the pipeline and turns `--batch-mtp` off. Only a layer split of two stages on two GPUs has been run.

Measured on a layer split of 2x RTX 3080 20 GB (220 W cap) with a Xeon E5-2696 v4, UD-Q4_K_XL, `"layer_split": "23"`,
`"parallel": 2`, `--spec 4`: two concurrent greedy decodes of 300 tokens, the two streams' tok/s added, median of
the measurements (10 rounds of two streams after each restart of the engine, 20 measurements per restart). The
engine was 0.1.41 with the rest of our production stack on it (the shared KV pool, and the adaptive expert tier
running in batch windows in the first two arms; the pipelined path of the third never adapts in windows), so this is
`--batch-mtp` on top of that, not this change alone on main.

| arm | aggregate tok/s | restarts, measurements | per window |
| --- | --- | --- | --- |
| `--batch-mtp`, one group | **89.2** | 3, 60 | 3.3 rows, 34.6 ms, 66-67% of the proposals accepted |
| no `--batch-mtp`, one serial group | 77.3 | 2, 40 | 2.0 rows, 23.7 ms |
| no `--batch-mtp`, `--batch-groups` unset (the 0.1.41 default: two pipelined groups of one slot) | 77.7 | 1, 20 | one row per group step |

That is +15.4% for `--batch-mtp` against serial windows without it (95% bootstrap interval of the ratio of medians
+12.5% to +17.9%) and +14.8% against the pipelined default; the two arms without it tie. One stream alone decodes the
same with and without it (81.7 and 82.0 tok/s), and the two slot drafters cost 35 to 110 expert-cache slots on the
last GPU (50 MiB of private state each).

The pipelined default is better in one place: a new prompt of about 48k tokens arriving while one stream decodes was
read at 2508 tok/s with pipelined groups and 1217 tok/s in serial windows, and the decoding stream finished sooner
(25.0 against 18.0 tok/s over its whole run); `--batch-mtp` does not change that. So it pays where concurrent decodes
dominate; with long reads arriving beside decodes the pipelined groups can be the better choice. Not measured: more
than two stages (the report on #1253 found no gain at four stages), more than two slots, sampled decoding, an
upstream-main binary (the pipelined arm is our build, started with the flags 0.1.41 resolves to).

Our measurement of `--batch-mtp` on a layer split with 2x R9700 and all experts in VRAM: 80.2 -> 62.7 tok/s (-21.9%, 0/5 pairs faster) against the pipelined default, so it pays only when the experts do not all fit in VRAM.

Earlier, on engine 0.1.40.3 with this change and the same two cards, two concurrent streams: UD-Q4_K_XL 39.9 to 43.5
tok/s without and 44.5 to 51.3 with `--batch-mtp` (2 runs per arm), and a small Coder IQ1_M test model with all
experts in VRAM 84.0 without and 101.2 with it, the greedy text of both streams identical.

Text: with the same expert cache on both arms (the slot drafters take VRAM, so the cache differs unless the
reserve is adjusted) and `--pcie-frac 0`, `--adapt-every 0`, the greedy text with `--batch-mtp` was equal to the text
without it for 5 short prompts in all 6 comparisons (one restart per arm). That is text equality, not a bit-exactness
proof, and a 25k-token prompt read while the other stream decodes was not repeatable between two identical streams
of the same arm.

With a layer split, the engine options go into the config's `args`:

```
"args": [ ..., "--batch", "8", "--batch-groups", "4", "--trim-stage-weights" ],
"layer_split": "12,24,36"
```

| Option | What it does |
| --- | --- |
| `"parallel": N` / `--batch N` / `--slots N` (2..8 normally) | up to N conversations have batch slots; more requests wait for a free slot. Each slot gets its own state (a session carved like the stage's own: GDN recurrence, QSA K/V and indexer, PLE history) on every GPU of the split. With grouped MTP, more than 8 slots can rotate through eight-row windows if memory permits. |
| `--batch-groups G` | with a layer split (default: auto, below): the N slots in G groups that flow through the GPUs as a pipeline (GPU k runs one group while GPU k+1 runs another). G must divide N. 1 = all slots in one window, GPU after GPU. |
| `--batch-groups auto` | the default on a layer split since 0.1.41 (give no `--batch-groups`): the engine pipelines one group per GPU stage (the most that divide the slots; 8 slots on 4 GPUs = 4 groups of 2) and says so (`INFO batch_groups=G`). `--batch-groups 1` turns it off (all slots in one window, GPU after GPU). Measured, 8 clients, total tok/s against one group: 4 x R9700 166 against 86, 2 GPUs 109 against 78, 3 GPUs 110 against 78. With `--adapt-async 1` and no `--batch-groups` the default is one group (the adaptive tier between batch windows does not run in pipelined groups) and the log says so; with `--batch-mtp` and no `--batch-groups` the pipeline wins and `--batch-mtp` is left off (`--batch-groups 1` or `STRATA_BATCH_MTP_SERIAL=1` keeps it); `--batch-groups G` or `--batch-groups auto` given with `--batch-mtp` pipelines the groups and turns `--batch-mtp` off, and given with `--adapt-async 1` pipelines the groups and turns `--adapt-async` off (the pipelined path does not adapt). |
| `--trim-stage-weights` | with an **explicit** `--layer-split` (e.g. `12,24,36`, not `auto`): every GPU loads only the dense weights of its own layers instead of the whole model's (the same as `STRATA_STAGE_TRIM=1`, PR #639). The VRAM this frees goes to the expert cache. Useful without `--batch` too. |

The engine never refuses a count it cannot run: it says so in its log and runs what it can - at most 8 slots by
default (a window holds 8 rows), as many as fit in VRAM, or none (one request at a time) when not two fit. The server reads
the count the engine reports (`INFO batch_slots=N`), and `GET /v1/status` says it (`concurrency.serving`).

### What a slot costs, and what setup recommends

Every slot's session takes VRAM that the expert cache would otherwise hold: 0.56 GiB at a 32K context with 8-bit
KV, more with a longer context unless the KV cache streams (`--kv-resident`: then only the attention's 32K window
stays in VRAM, and each slot's whole KV cache takes pinned RAM - 1.6 GB at 128K). On a card whose experts mostly run
on the CPU, a batch also reads about as many distinct experts as the requests one by one (different conversations
route to different experts), so the gain is in **latency** (nobody waits for a whole answer), and a request alone
runs slower (the smaller expert cache): 11-24 % on a 12 GB card, see the measurements below.

So setup recommends `"parallel"` **only where the experts mostly fit in VRAM**: the expert cache (each card's VRAM
less ~5 GB, every card of a layer split counted) must still hold at least half of the model's experts beside the
slots, and the slots may take at most a fifth of it, up to 4 slots. With Q2_0 at 32K that is 3 slots on a 24 GB
card, 4 from 32 GB or on a split such as 2 x 16 GB; IQ3_S needs 32 GB or a split. Everywhere else (any 12 or 16 GB
card alone) it stays at one at a time and setup says: "parallel N reduces waiting for several users but costs
about 10-25% speed per request on this card". `--parallel N` is honoured as asked either way.

## How the server uses the slots

- **One request alone** runs on the usual solo path (verify windows with MTP drafts): the fastest single stream.
- **When a second request arrives**, the first is stopped (`STOP`) and continues in a batch slot with its prompt
  plus what it generated so far - the engine's prompt cache holds exactly that, so nothing is read again - and the
  new request is admitted next to it. By default, a request in a slot decodes **without MTP drafts** (one token per window).
  With `--batch-mtp`, each slot verifies one MTP proposal alongside its current token.
- **A request left alone in a slot** (the others finished, nobody waits) goes back to the solo path: the slot is
  stopped, the engine copies its sessions back and decodes with MTP drafts again (at most twice per request; with
  `--prompt-cache 0` it stays in the slot; `STRATA_PARALLEL_SOLO=0` turns it off). The draft layer's own K/V was
  built for another conversation then, but measured it accepted as many drafts (140 of 172) as a draft layer that
  read the conversation (140 of 173).
- **More requests than slots** wait for a free one (`/metrics` -> `live.slots` shows each slot: idle, reading or
  decoding, its tokens and tok/s; `live.running` the requests in flight).
- **Each admission** reads the request's prompt through the usual prompt path (prompt cache and conversation
  checkpoints included) and produces its first token there; the state is then copied into the slot. Admissions
  are taken one at a time, and **the slots decode between the prompt's chunks**: after each chunk (`--prefill`,
  2048-8192 tokens) they decode for half as long as the chunk took (`STRATA_BATCH_DECODE_SHARE`, default 0.5), so
  a long prompt slows the others down instead of stopping them. The chunks are the ones one uninterrupted read
  takes, so the prompt's arithmetic is unchanged.
- **A long prompt gives way to a short one** (#656's cooperative preemption): when a request with a prompt under
  half as long is waiting, the server sends `BYIELD`; at its next chunk boundary the long read stops, the part read
  so far is copied into a slot, the short request is admitted, and the long one then goes on from its slot with the
  same chunks (at most twice per request).
- **Each slot is a conversation cache.** A finished slot keeps what it holds (the prompt, the answer, and the
  checkpoint at the prompt's last turn boundary); the next turn of that conversation goes to that slot and the
  engine copies its state back (50-60 ms for a short conversation) instead of reading the history again - also for
  a client that drops the reply's thinking from the history (the checkpoint matches up to the new turn). A new
  conversation takes an empty slot, else the one used longest ago.
- Slots are assigned so that consecutive requests land in different pipeline groups (`--batch-groups`).
- A client that disconnects stops its slot (`BSTOP`); the others go on.

## The VRAM expert tier keeps adapting in batch windows

Batch windows count which experts they route to, and every `--adapt-every` windows (default 4) the engine swaps the
most-routed experts that are not in VRAM into the cache in place of the least-routed resident ones, the same rule the
solo path uses (`--adapt-swaps` and `--adapt-decay` apply too; on a layer split every GPU's cache adapts). It runs when the
tier is not the whole model and both `--adapt-every` and `--adapt-swaps` are above 0. By default the round runs beside the
window's commit and drafts and has landed before the next window starts. It stands still while a long prompt is read
between decoding windows (the prompt may hold a loan of cache slots). The `strata batch:` log line shows the rounds and swaps, each stage's
GPU-reach wait and pool time, and how many routed entries the VRAM tier, the PCIe share and the CPU served per window.
Swaps move experts between the GPU and the CPU, which round differently, so greedy outputs can differ from run to run;
`--adapt-every 1000000` keeps the tier fixed (as in the exactness settings below). The pipelined `--batch-groups` path
does not adapt.

With `--adapt-async 1` (needs the resident RAM mode) the round does not hold a window up: each batch window moves the
round on one step (table changes and uploads on the main thread, the copies on a helper thread and the cards' refill
streams), as on the solo path. A request that arrives finishes the round in flight before it starts, and a prompt read
between decoding windows does not move it. `--adapt-min-gain F` (default 1.5) raises the bar a swap must clear (the
routing count of the expert coming in against the one going out), so fewer experts move per round; both tiers use it.
The log line's `adapt wait` is the time the main thread spent on the tier per window (the blocking tier: waiting for
its round).

Measured with the same setup as the `--batch-mtp` numbers above (2x RTX 3080 20 GB, UD-Q4_K_XL, `"layer_split": "23"`,
`"parallel": 2`, `--batch-mtp`, engine 0.1.41 with our production stack; two concurrent greedy decodes, the streams'
tok/s added, median of 20 measurements per restart):

| tier | aggregate tok/s | restarts | per window | routed entries served from VRAM |
| --- | --- | --- | --- | --- |
| `--adapt-async 1 --adapt-every 2` | **89.2** | 3 | 34.6 ms, adapt wait 0.2 ms | 92.1% |
| blocking, `--adapt-every 2` | 74.7 | 1 | 41.4 ms, adapt wait 8.4 ms | 92.3% |
| none, the cache frozen from a profile the tier had learned on the same prompts | 69.5 | 1 | 43.6 ms | 77.2% |
| none, `--adapt-every 0` | 45.2 | 2 | 66-68 ms | 52.7% |

The switch changes the tier in the solo path as well (there is no switch for batch windows alone), so the table
compares the tier as a whole, not "batch windows only": the solo decode speed moves the same way (81.7, 66.3, 69.0 and
43.1 tok/s). The prompts are the benchmark's own, eight topics that repeat, which the tier learns quickly, so on a
more varied workload the gain may be smaller (not measured). Upstream main runs no tier in batch windows, and this
was not measured against an upstream binary.

## Exactness

A batch row's arithmetic is the single-token window's, so with greedy decoding **every conversation of a batch
produces exactly the tokens it produces alone** - verified token by token for 8 concurrent conversations of 150
tokens, with and without the pipeline (`tools/batch_test.py`), and on one RTX 5070 for 4 conversations, for a long
prompt read while two others decode, for a prompt that gave way and went on, for a next turn continued from its
slot (from all it holds, and from its turn checkpoint without the reply's thinking), and for a request stopped in
its slot and continued on the solo path (`tools/batch_interleave_test.py`). These
settings make the comparison exact:

- `STRATA_IQ_MT_MIN=1` (the multi-token CPU expert kernels for every group, as for the solo path's own
  exactness tests: by default an expert's rows round differently alone than in a group, so the output depends on
  how many rows of a window share an expert - which differs between a batch and a request alone),
- `--pcie-frac 0`: the PCIe share of the missed experts is chosen per window from the window's misses, so the
  same expert can run on the GPU in one window and on the CPU in another, which rounds differently, and
- `--adapt-every 1000000` (the VRAM tier fixed), and `--no-prefill-borrow` while a prompt is read beside decoding
  slots: their windows then see the expert cache without the slots the prompt borrowed (those experts run on the
  CPU).

With the default settings the outputs stay coherent but drift apart after some tokens, as two solo runs whose
windows differ can. Measured on the RTX 5070 (Q2_0, 4 conversations of 200 tokens, `tools/batch_test.py` without
the settings above): one equal to its solo run, the others apart from token 0, 99 and 174 - the first token already
differed for one, because the adaptive VRAM tier had moved experts between the solo runs and the batch; all four
texts read as well as their solo runs.

Sampled requests (temperature, top_p, top_k, min_p, seed) are drawn row by row with the solo window's
counter-based draw (Philox(seed, position)).

## Limits (for now)

- By default, batch windows carry no MTP drafts: a conversation in a slot decodes one token per window (the solo
  path keeps its drafts, which is why a request alone is not put in a slot, and goes back to it when left alone).
- `--batch-mtp` uses one proposal per slot. With a layer split it needs each stage on its own GPU, does not combine
  with `--batch-groups` above 1 (pipelined slot groups run no drafts, #1413) and has not been run with helper expert
  caches.
- Repetition / frequency / presence penalties are not applied in batch windows.
- A prompt shorter than one chunk is read in one piece (the slots wait for it); a read gives way only at a chunk
  boundary, and not for pictures.
- Admissions are one at a time: two new long prompts are read one after the other.
- `--batch-groups` needs every stage on its own GPU. A pipelined slot is a conversation cache again (0.1.41): a request left alone in its slot goes back to the solo path with its drafts, as on one GPU.
- The slot sessions take VRAM (above) and, with KV streaming, pinned RAM.

## Measured

One RTX 5070 (12 GB), Ryzen 5 7600, 64 GB DDR5, Q2_0, 32K context, through the HTTP server: C different requests
sent at once (an 800-word essay each, 256 tokens per answer, greedy, thinking off), median of 3 rounds:

| Concurrent | Setting | Total tok/s | vs one at a time | Per request tok/s | First token: median / last of the round |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | one at a time (default; what setup recommends on this card) | 74.3 | - | 83.6 | 0.4 s / 0.4 s |
| 1 | `"parallel": 2` | 67.3 | -9 % | 74.8 | 0.4 s / 0.4 s |
| 1 | `"parallel": 4` | 57.9 | -22 % | 63.8 | 0.4 s / 0.4 s |
| 2 | one at a time | 71.6 | - | 77.3 | 2.1 s / 4.1 s |
| 2 | `"parallel": 2` | 61.0 | -15 % | 32.6 | 0.5 s / 0.9 s |
| 2 | `"parallel": 4` | 54.6 | -24 % | 28.7 | 0.6 s / 0.7 s |
| 4 | one at a time | 70.7 | - | 79.6 | 6.0 s / 11.2 s |
| 4 | `"parallel": 2` | 61.2 | -13 % | 32.2 | 4.5 s / 9.3 s |
| 4 | `"parallel": 4` | 63.1 | -11 % | 16.9 | 1.0 s / 1.8 s |

On this card the slots buy **waiting time, not speed**: the fourth of four requests starts after 1.8 s instead of
11.2 s, but together they decode 11-24 % slower than one after the other, and a request alone loses 11 % (2 slots)
or 24 % (4 slots), because the slots' sessions (0.56 GiB each) come out of the expert cache and most experts run
on the CPU: a batch window over 4 conversations reads 24 CPU experts per layer against ~8 for one, so it costs about
what the 4 tokens cost one after the other (`strata batch:` in the engine log: 54 ms per 4-row window, ~20 ms per
1-row window). This is why setup leaves a 12 GB card at one at a time. Cards that hold most experts in VRAM, and a
layer split, are where the slots also add speed (below).

A 4-GPU layer split (4 x 16 GB, PCIe Gen3), IQ3_S, `--batch 8 --batch-groups 4 --trim-stage-weights`, through
the HTTP server, 400 tokens per answer, temperature 0.7 (PR #559):

| Concurrent requests | Per request | Total |
| ---: | ---: | ---: |
| 1 | 123 tok/s (solo path) | 120 tok/s |
| 2 | 57 tok/s | 113 tok/s |
| 4 | 51 tok/s | 205 tok/s |
| 8 | 45 tok/s | 360 tok/s |

With the patches below on engine 0.1.38 and parking on, through the service: 8 requests at temperature 0 -> 369
tok/s, at 0.7 -> 358 tok/s.

`--trim-stage-weights` alone raised the share of experts held in VRAM on that machine from 76-85 % to 84-100 %
per card.

## Together with conversation parking

`--conversation-cache-mib N --conversation-cache-slots K` (DETAILS.md) works with the layer split too: a request
whose conversation was parked is restored on every stage before its admission, so an agent and its sub-agents, or
several chats that alternate, come back without reading their history again. Measured on the same 4-GPU split,
two long conversations alternating through the HTTP server: the first turns took 3.9 s and 5.7 s to the first token
(their prompts read), the follow-ups 0.53 s and 0.46 s.

## Testing

Four scripts drive a built engine or a running server; each exits non-zero on a failure. `serve/test_parallel.py`
tests the server's side with a scripted engine (no GPU).

| Script | What it checks |
| --- | --- |
| `tools/batch_test.py` | the same prompts alone (`GEN`) and together in the batch slots (`BGEN`): every slot's greedy tokens equal its solo tokens; prints the aggregate rate. `--batch-groups` in `--extra` tests the pipeline, `--keys "temperature=0.7"` the sampled rows. |
| `tools/batch_interleave_test.py` | a long prompt read while two slots decode, a prompt that gives way (`BYIELD`) and goes on, and a next turn continued from its slot: each equal to its solo tokens. |
| `tools/parking_test.py` | a follow-up to a conversation decodes the same tokens whether its state stayed live or came back from the parking cache (with a layer split: every stage's image). |
| `tools/early_close_test.py` | a client that stops reading a streamed answer early (alone, and with a second request running) does not leave its tokens to the next request (server). |

For exact comparisons pass `--pcie-frac 0 --adapt-every 1000000` (and the scripts set `STRATA_IQ_MT_MIN=1`):

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 8 --n 8 \
    --extra "--layer-split 12,24,36 --trim-stage-weights --batch-groups 4 --pcie-frac 0 --adapt-every 1000000"
python3 tools/batch_interleave_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
python3 tools/parking_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--layer-split 12,24,36 --conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0"
STRATA_KEY=<key> python3 tools/early_close_test.py http://127.0.0.1:8080
```

For single-GPU `--batch-mtp`, use a model with `rt/draft_vocab.bin` to exercise admission with a
shared draft-vocabulary head. This compares both slots' output against solo decoding:

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 2 --n 2 --max-new 64 \
    --extra "--batch-mtp --pcie-frac 0 --adapt-every 1000000"
```

## DeepSeek-V4.1 (`ds41_generate`)

The V4.1 engine speaks the same slot protocol (the table below). Add `"parallel": N` to a ds41 config (the
DeepSeek-V4.1 server passes `--batch N`; by hand: `--batch-slots N`, or `--batch-slots auto`). The slots need the
conversation pool (`--conversation-cache-mib`, on by default). Without the option the engine is the one-slot engine
it always was, bit for bit.

**What a slot is.** Each slot holds a whole conversation of the full `--ctx`: the window rings and their snapshot, the
ratio-2 compressor states, the compressed KV and the indexer keys, the turn snapshot, the park and the Engram history.
With `--kv-fp4` a slot is 234 MiB at 256K (0.46 GiB at 512K, 0.91 GiB at 1M). That VRAM comes out of the expert cache,
about 67 experts per GiB. `--batch-slots auto` takes as many slots as 2 GiB holds (`STRATA_BATCH_SLOTS_GIB`), from 2 to
8. The engine prints `batch slots: N x ... MiB` when it starts.

**A batch window.** A window carries one token of every decoding slot, plus rows of a short prompt being read
(`STRATA_SERVE_STEP_MAX`, 96 in the shipped wrapper), at most 16 rows in all. The window goes through every layer in
one pass:

- the projections, hyper-connections, router, shared expert and head run over all rows;
- the rings, compressed rows and indexer keys of all slots of a layer live in one tensor (a slab). So the ring
  write, the window gather, the compressor, the compressed-row gather and the flash-attn run once for all rows. Only
  the indexer ranks each slot's rows separately;
- the routed experts run once per layer over the union of the rows (`run_multi_to_dev`).

A request alone runs the single-sequence path with DSpark, exactly as without slots.

**Exactness.** On the CPU backend a window's rows are bit for bit each conversation's solo rows (`test_ds41_batch`:
40 layers, 3 conversations, 8 steps, and a prompt read inside the windows). On the GPU the dense matmuls run on more columns, so the bits differ from solo. The difference stays inside the noise
of the solo path itself (`test_ds41_batch --backend cuda`, 3 conversations x 8 steps). Batch windows against solo
steps: top-1 23 of 24, KL mean 3.9e-3, max 1.2e-2. A solo 8-row pass (what a DSpark verify pass is) against the same
steps: 22 of 24, KL mean 4.4e-3, max 1.7e-2. The slabs and the merged flash-attn change nothing in that: without them
the windows give the same numbers. The solo path keeps its bits on the GPU. Its serve text set gives the same logit
dumps as the engine without slots, and a slot moved by `swap_slots` or restored from the pool into another slot
decodes bit for bit as before.

**Measured** (RTX 5090, EPYC with 624 GB RAM, heretic 1017 Q3_K experts in pinned RAM, exact routing, `--kv-fp4`,
`--tier-batch-pcie 0.1`, 48 threads, 8 slots of 256K). Each conversation is a ~1,500-token text plus growing turns.
Each request asks a short new question (~40 new tokens) and decodes 200 greedy tokens, all N at once. "Steady" means
the tokens decoded while all N were running:

| Concurrent | Steady total tok/s | Per request tok/s | First token (mean / max) | Window (rows) |
| ---: | ---: | ---: | ---: | --- |
| 1 (solo, DSpark) | 48-51 | 48-51 | 0.4 s | - |
| 2 | 62.5-63 | 30.5 | 0.6-0.7 / 0.9 s | 30.5 ms (2) |
| 4 | 74-81 | 18-19.5 | 1.1-1.2 / 1.9 s | 47.6 ms (4) |
| 8 | 94-95 | 11.6 | 2.4-2.6 / 4.8 s | 75.1 ms (7.8): dense 23.3 + experts 50.0 |

With the same prompts as their solo turns (nothing to read), 2 / 4 / 8 conversations decoded at 65 / 82 / 103 tok/s.
That matches a cost model fitted on the production log before the work: 62 / 82 / 103, against 50 tok/s one at a
time. Without the slabs (`STRATA_SLOT_SLAB=0`, each slot's attention on its own tensors) a window of 8 rows takes
90.5 ms instead of 78. The experts' part
follows the cost of the unique experts per window; the rows of different conversations share few experts (about 27%
at 8 rows), so the gain comes from sharing the fixed part of a pass.

**Limits.** DSpark runs only for a request alone. `-exact` and `-fast` requests never share a window, because the
route bias is global. Pictures are read alone, and their decode joins the windows. A prompt with more than 96 new
tokens is read layer-major while the windows wait.

**Testing.** `test_ds41_batch -m M --engram A:B --prompts p0.i32,p1.i32,p2.i32 [--backend cuda --kv-fp4]` checks
the engine core: solo against the reference path and the batched windows, a prompt read in the windows, a slot moved
by `swap_slots`, a conversation saved from one slot and restored into another, and the noise floor of a multi-row solo
pass. `DS41_TEST_LAYERS=n` runs the first n layers only. `tools/v41/engine/check_batch_slots.py` checks the protocol
against one-at-a-time references.

## Engine protocol (`--serve`)

On top of `GEN` / `GENI`:

| Line | Direction | Meaning |
| --- | --- | --- |
| `BGEN <slot> <max_new> [keys] <ids>` | in | read the prompt (as `GEN 1`), then continue in `<slot>` |
| `BGENI <slot> <max_new> [keys] <file> <ids>` | in | the same with images |
| `BADM <slot> <1/0>` | out | after the admission's `DONE`: 1 = it continues in the slot, 0 = it ended |
| `BT <slot> <id>` | out | a token of that slot |
| `BDONE <slot> <generated> <stop/length/cancel> <ms>` | out | the slot is free again (it keeps its conversation) |
| `BSTOP <slot>` | in | end that slot at its next window |
| `BYIELD <slot>` | in | the prompt being read gives way at its next chunk boundary; its part read waits in `<slot>` (the admission's own, or a free slot for a solo request) |
| `YIELDED <slot> <tokens>` | out | before the `DONE cancel` of a read that gave way: the request is sent again later and goes on from there |
| `INFO ... batch_slots=N` | out | the slots the engine runs (only with `--batch`) |

`tools/batch_test.py` drives the engine directly: the same prompts alone, then together, compared token by token,
and the aggregate rate.
