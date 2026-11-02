#!/usr/bin/env python3
"""parity.py REF.f32 OURS.f32 N_VOCAB - xolotl-lfm3 --dump against lfm3_ref_dump (llama.cpp): per-position KL(ref||ours),
same argmax, max |logit diff|.  Gate for P1: mean KL < 1e-3 and >= 99% same argmax on the same GGUF."""
import sys
import numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32); b = np.fromfile(sys.argv[2], dtype=np.float32); v = int(sys.argv[3])
a = a.reshape(-1, v); b = b.reshape(-1, v); n = min(len(a), len(b)); a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
la = a - a.max(1, keepdims=True); la -= np.log(np.exp(la).sum(1, keepdims=True))
lb = b - b.max(1, keepdims=True); lb -= np.log(np.exp(lb).sum(1, keepdims=True))
kl = (np.exp(la) * (la - lb)).sum(1)
same = (a.argmax(1) == b.argmax(1)).mean()
print(f"positions {n}  mean KL {kl.mean():.3e}  p99 KL {np.percentile(kl, 99):.3e}  max KL {kl.max():.3e}  "
      f"same argmax {100 * same:.2f}%  max |dlogit| {np.abs(a - b).max():.3f}")
for lo, hi in ((0, 256), (256, 1024), (1024, n)):
    if hi > lo and lo < n:
        print(f"  pos {lo}-{min(hi, n)}: KL {kl[lo:hi].mean():.3e}, same argmax {100 * (a[lo:hi].argmax(1) == b[lo:hi].argmax(1)).mean():.2f}%")
