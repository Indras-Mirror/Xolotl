#!/usr/bin/env python3
"""parity.py REF.f32 OURS.f32 N_VOCAB [STRIDE] [N_TOKENS] - xolotl-glimmer --dump against glimmer_ref_dump (llama.cpp):
per-position KL(ref||ours), same argmax, max |logit diff|.  Both files hold every STRIDE-th position plus the last.
Gate for P1: mean KL < 1e-3 and >= 99% same argmax on the same GGUF, before and after the sliding ring wraps
(window 2048 + ubatch 512 -> 2560 cells)."""
import sys
import numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32); b = np.fromfile(sys.argv[2], dtype=np.float32); v = int(sys.argv[3])
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
a = a.reshape(-1, v); b = b.reshape(-1, v); n = min(len(a), len(b)); a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
pos = np.arange(n) * stride
if len(sys.argv) > 5:
    pos[-1] = int(sys.argv[5]) - 1
la = a - a.max(1, keepdims=True); la -= np.log(np.exp(la).sum(1, keepdims=True))
lb = b - b.max(1, keepdims=True); lb -= np.log(np.exp(lb).sum(1, keepdims=True))
kl = (np.exp(la) * (la - lb)).sum(1)
same = a.argmax(1) == b.argmax(1)
print(f"positions {n} (stride {stride})  mean KL {kl.mean():.3e}  p99 KL {np.percentile(kl, 99):.3e}  max KL {kl.max():.3e}  "
      f"same argmax {100 * same.mean():.2f}%  max |dlogit| {np.abs(a - b).max():.3f}")
for lo, hi, what in ((0, 2048, "inside the window"), (2048, 2560, "window full, ring not wrapped"), (2560, 1 << 30, "ring wrapped")):
    sel = (pos >= lo) & (pos < hi)
    if sel.any():
        print(f"  pos {lo}-{min(hi, pos.max() + 1)} ({what}): KL {kl[sel].mean():.3e}, same argmax {100 * same[sel].mean():.2f}%")
