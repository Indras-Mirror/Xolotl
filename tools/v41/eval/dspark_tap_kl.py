#!/usr/bin/env python3
"""tools/v41/eval/dspark_tap_kl.py - how far a DSpark run's verify logits are from the exact prefill of its own stream.

STRATA_DSPARK_DUMP writes every verify row as {int32 pos, m, row, refeed, nv floats}; row `pos` predicts token pos+1,
so it lines up with row `pos` of `ds41_generate --dump-logits` run over the same token stream (prompt + the run's own
output).  KL(prefill || verify) and top-1 agreement over every verify row measure what a decode-side approximation
(e.g. STRATA_FILE_DEFER) costs; the same numbers for a run without it are the engine's own floor.

    dspark_tap_kl.py DSPARK_DUMP.bin PREFILL_LOGITS.f32 N_STREAM_TOKENS
"""
import sys
import numpy as np

dump, pre, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
P = np.fromfile(pre, dtype=np.float32)
nv = P.size // n
assert P.size == n * nv, 'prefill logits are not n x vocab'
P = P.reshape(n, nv)
raw = np.fromfile(dump, dtype=np.uint8)
rec = 16 + 4 * nv
assert raw.size % rec == 0, f'dump size {raw.size} is not a multiple of {rec}'
R = raw.reshape(-1, rec)
hdr = R[:, :16].copy().view(np.int32).reshape(-1, 4)
# A verify row past a rejected draft was computed on the DRAFT tokens, not the stream's, so it has no prefill row to
# match.  The last write of each position is the one whose inputs are the final stream's (a rejected position is
# verified again after the correction), so only that one is scored.  (2026-10-23: scoring every row gave a nonsense
# floor - KL 4.17, top-1 63 % for the run WITHOUT any approximation.)
last = {}
for i in range(len(R)):
    pos = int(hdr[i, 0])
    if 0 <= pos < n - 1:
        last[pos] = i
kl = agree = cnt = 0.0
for pos, i in sorted(last.items()):
    q = R[i, 16:].copy().view(np.float32).astype(np.float64)
    p = P[pos].astype(np.float64)
    lp = p - (np.log(np.exp(p - p.max()).sum()) + p.max())
    lq = q - (np.log(np.exp(q - q.max()).sum()) + q.max())
    kl += float((np.exp(lp) * (lp - lq)).sum())
    agree += float(p.argmax() == q.argmax())
    cnt += 1
print(f'{{"rows": {int(cnt)}, "kl": {kl / cnt:.6f}, "top1_agree": {agree / cnt:.4f}}}')
