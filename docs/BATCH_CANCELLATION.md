# Batch cancellation channel

Set `STRATA_BATCH_CANCEL_PIPE=1` in the Python server's environment to enable
request-owned batch cancellation. It is off by default. Leave it unset to keep
the existing control protocol and cancellation route.

On POSIX systems the server opens a separate inherited pipe for small cancellation
messages. This avoids waiting behind a large prompt written to the engine's stdin.
The engine advertises `batch_cancel_epoch=1` and `batch_cancel_fd=1` in its INFO
line. Each admission carries a monotonically increasing cancellation epoch;
`BSTOP <slot> <epoch>` only cancels that owner, including before admission starts.
A stale request cannot stop a later user of the same slot.

A full or unavailable pipe falls back to epoch-tagged stdin, which can still wait
behind a prompt write. Windows uses this stdin fallback. Engines without the epoch
capability retain the existing untagged protocol. The direct native protocol uses
`STRATA_CANCEL_FD` only for the inherited POSIX read descriptor, with the feature
enabled; users should normally let the Python server create it.

Readers publish atomic cancellation requests only. The scheduler releases an active
slot at a committed boundary. During admission it can stop between existing
completed checkpoint/KV copy operations; a partially written target is excluded
from cache reuse. No new GPU synchronization is added. A single copy, fused kernel,
or multi-GPU pipeline can still delay cancellation. Slot inactivity does not mean
that all model allocations have been freed, and this is not a fixed latency bound.

CPU regressions (no model required):

```sh
python -m unittest serve.test_batch_cancel_pipe
python tools/test_batch_cancel_copy.py
python tools/test_batch_cancel_protocol.py
c++ -std=c++20 -pthread -Iinclude src/program/batch_cancel_test.cpp -o batch_cancel_test
./batch_cancel_test
```

The copy test extracts the actual admission helper and transition, substitutes
completed device operations, and checks cancellation, errors and asynchronous
scratch lifetime with AddressSanitizer/UndefinedBehaviorSanitizer. It requires a
compiler with these sanitizers (`CXX` can select one). It does not validate GPU
execution or multi-GPU timing.
