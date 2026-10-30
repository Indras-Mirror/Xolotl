# Deep sleep: POST /sleep and /wake

> **Status: draft (merged 2026-10-31 for testing, #75).** The GPU measurements below predate the rebase onto current
> main. On the merged tree only the model-free checks were run (39/39 deep-sleep tests, the rest of serve/ passing); the
> DeepSeek-V4.1 checkpoint/restore and the M3.1 native wake still need a GPU run on current main before this is
> considered done. Off by default, so normal serving is unaffected. Tracking: #74.

A sleeping engine keeps its model and conversation state in host RAM while
releasing its device resources. It must be put to sleep explicitly: ordinary
requests do not trigger sleep or a new idle policy. The next request wakes a
sleeping engine, or a controller can call `/wake` first. Prefix reuse remains
subject to the engine's normal matching policy.

## HTTP control

Enable the feature with `"deep_sleep": "auto"` in the server config first; it is
off by default. Use the same API key, Host/Origin rules and JSON content type as `/unload`:

```sh
curl -X POST -H 'Content-Type: application/json' http://127.0.0.1:8080/sleep
curl -X POST -H 'Content-Type: application/json' http://127.0.0.1:8080/wake
```

| Condition | Status |
| --- | --- |
| sleeping, or already asleep | 200, `status: asleep` |
| awake after `/wake`, or already awake | 200 |
| running/queued request, busy batch slot or admission | 409, `status: busy` |
| unavailable method, or `deep_sleep: off` | 501, `status: unsupported` |
| model not loaded | 200, `status: not loaded` |
| restore needs more free device memory | 503 with `Retry-After: 15`; remains asleep |

`/sleep?wait=1` waits for a request turn, up to 300 seconds; `?wait=<seconds>`
sets an explicit limit. `/health` and `/v1/status` report `asleep`; `/props`
sets `is_sleeping`. `/sleep` reports the chosen method and measured phase times.

A wake follows load admission: `before_load`, then `min_free_vram_mib`, then
restoration. A vision helper is checkpointed before the engine so it gives back
its VRAM too; if unsupported, the helper is stopped and can restart with the next
picture. An elastic expert cache releases its helper reservation as usual.
Checkpointed helpers are restored after the engine. A pending VRAM reserve is
applied after wake. `/slots` operations also wake the engine.

Unloading or restarting an asleep engine ends its checkpointed descendants as
well. Tracked process birth times are checked before driver operations; cleanup
uses process descriptors for signalling where the OS supports them. An exited
process's reused PID is not treated as the saved engine or helper. A partially
unlocked group is fatal: the tracked processes are ended instead of being
advertised as safely asleep. A retryable GPU-memory refusal retains the checkpoint.

## Methods and configuration

Set `deep_sleep` in the server config to `auto`, `native`, `checkpoint` or `off`
(default). `auto` prefers native capability and otherwise tries the CUDA driver.
No CUDA checkpoint API is loaded until it is needed. Unsupported drivers and
non-CUDA platforms report unsupported; they do not silently unload the model.

### Native SLEEP/WAKE

The engine advertises `sleep=1` on an `INFO` line before `READY`. Between
requests it implements:

```text
SLEEP -> SLEPT seconds=... kv_tokens=... conversation=0|1 ...
WAKE  -> INFO wake_seconds=...       (optional)
         READY <context> stop
```

`ERR` after `SLEEP` means the engine stayed awake. `ERR` after `WAKE` means it
remains asleep. Repeated sleep/wake is idempotent. A direct pipe client can also
send a generation request while asleep; the engine wakes before generation.
MiniMax-M3.1 implements native sleep on CUDA. Its retained buffers, arena startup
settings and measured limits are in [MINIMAX_M31.md](MINIMAX_M31.md#deep-sleep).

### CUDA process checkpoint

On Linux, the server checks whether the NVIDIA driver exports
`cuCheckpointProcessGetState`, `Lock`, `Checkpoint`, `Restore` and `Unlock`.
It checkpoints the engine and its CUDA descendants, allowing a launcher process
that does not `exec` its child. The driver copies device allocations to host RAM,
releases device resources and restores the same addresses. Enough host RAM and
free VRAM for the original allocations are required.

All tracked processes must be checkpointed or rolled back together. Failed
checkpoint rollback or partial unlock ends that group; a later request starts a
new engine. Only the captured process group is cleaned up. A CUDA call attempted
by an asleep process waits for restoration, so unloading cannot depend on its
normal `QUIT` handling.

Engines advertising `checkpoint_clock=1` additionally negotiate:

```text
CHECKPOINT_PAUSE  -> CHECKPOINT_PAUSED
CHECKPOINT_RESUME -> CHECKPOINT_RESUMED paused_us=<integer> graph_execs=<integer>
```

DeepSeek-V4.1 uses this to exclude the driver-checkpoint interval from CUDA graph
cache eviction ages. The engine pauses before checkpoint and resumes only after
restore/unlock, including a successful checkpoint rollback. It preserves captures,
KV, conversations and expert caches. It does not exclude ordinary idle time or
refresh graphs already old before the pause. Invalid or missing acknowledgements
are fatal; a retryable driver-restore failure keeps the pause in place.

## Validation and limits

Earlier controlled Linux/CUDA runs used an EPYC 9654, 624 GB RAM and one RTX 5090
32 GB. Results apply to those configurations, not every model or driver:

- MiniMax-M3.1: numerical boundary/gate20 fixtures, growing conversations and
  repeated sleep/wake passed; a focused five-request vision/text comparison
  matched answers and planned chunks. Native wake was about 1.4–1.9 s. Its first
  4K prefill was 3.5–4.1% slower after wake; the second prefill was essentially
  neutral. The chosen 3% first-prefill tolerance was **not met**. The first picture
  request after wake took 15.38 s in one focused run. Wake time alone does not
  promise that response latency.
- DeepSeek-V4.1 with default CUDA graphs/fusion: a serial vision checkpoint A/B
  matched four responses, eight nonempty logit dumps and two embedding files.
  A separate growing text conversation matched three responses and three logit
  dumps, retaining 678 cached tokens before sleep and 700 after wake. Driver
  restore took about 18.6–20.9 s. The image fixture itself reported zero cached
  tokens; it does not establish an image conversation-cache hit.
- Earlier DeepSeek tests without checkpoint-clock handling diverged after sleep;
  an equally long idle control could reproduce the divergence without checkpoint.
  Clearing graph caches or disabling keep-cache did not fix that control. The
  metadata-clock adjustment passed the focused serial tests; this does not prove
  a particular driver corruption bug or parallel determinism.
- Earlier serial Strata text/image comparisons matched logit dumps; parallel
  answers differed. Dense Q27 answers matched, but no logit dumps were collected.
  These are limited observations, not general numerical qualification.

The final patch restores the previously tested computational path after removing
ineffective resource-preload experiments. CPU/CUDA rebuilds and model-free tests
are separate from GPU numerical/performance evidence, which predates this rebase.
No hidden model warmup or eager-loading policy is added. HIP, SYCL, Windows,
other GPUs and broad concurrent checkpoint scenarios were not validated.

## Reproducing the checks

Model-free protocol, rollback, PID-reuse and helper-lifecycle tests:

```sh
python -m unittest serve.test_deep_sleep serve.test_vision_lazy \
  serve.test_vision_busy serve.test_vision_elastic
python tools/m31/tests/check_registered_arena_queue.py --out build/queue-check
python tools/xolotl/ggml/check_checkpoint_clock.py --out build/clock-check
```

Build the normal pinned dependency using `bash tools/xolotl/fetch-llama.sh`, then
follow the [M3.1 build instructions](MINIMAX_M31.md) or the existing DeepSeek build
instructions. No private patched checkout is needed. To reproduce a hardware
comparison, start two fresh processes with the same model and generation settings;
compare a sequence of requests with and without sleep between the same turns.
Use equal idle intervals, check actual `cached_tokens` and prompt chunks, compare
nonempty numerical dumps where supported, and measure both `/wake` and the first
following response. Stop other work before a timing run. Preserve failures and
report untested backends rather than inferring parity from equal answer text.
