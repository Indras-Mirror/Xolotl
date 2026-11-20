#!/usr/bin/env python3
"""Numpy reference for test_moe_train / tools/train/moe_train.hpp (QFN MoE: softmax gate, top-K, weight norm, SILU).

usage: moe_ref.py dump.bin

Recomputes the same forward from the dumped inputs and compares with the C++ graph output, so the gate checks that the
graph IS the documented computation (the FD check alone only proves it is self-consistent).
"""
import sys

import numpy as np


def main(path):
    raw = open(path, "rb").read()
    D, F, E, K, T = [int(v) for v in np.frombuffer(raw[:20], dtype=np.int32)]
    off = 20
    sizes = [(D, T), (D, E), (D, F, E), (D, F, E), (F, D, E), (D, T), (D, T)]
    arrs = []
    for ne in sizes:
        n = int(np.prod(ne)) * 4
        arrs.append(np.frombuffer(raw[off:off + n], dtype=np.float32).reshape(ne[::-1]).transpose(
            *range(len(ne) - 1, -1, -1)).astype(np.float64))
        off += n
    x, router, up, gate, down, r, o_ref = arrs
    wts = np.frombuffer(raw[off:off + K * T * 4], dtype=np.float32).astype(np.float64)
    off += K * T * 4
    ids = np.frombuffer(raw[off:off + K * T * 4], dtype=np.int32)

    logits = router.T @ x                                        # [E, T]
    p = np.exp(logits - logits.max(axis=0, keepdims=True))
    probs = p / p.sum(axis=0, keepdims=True)
    sel = probs[ids.reshape(T, K).T, np.arange(T)[None, :]].T    # [T, K] softmax prob of each selection
    w = sel / np.maximum(sel.sum(axis=1, keepdims=True), 6.103515625e-5)

    o = np.zeros((D, T))
    for t in range(T):
        for k in range(K):
            e = int(ids[t * K + k])
            v = gate[:, :, e].T @ x[:, t]
            a = v / (1.0 + np.exp(-v))                            # silu(x) = x * sigmoid(x)
            h = a * (up[:, :, e].T @ x[:, t])
            o[:, t] += w[t, k] * (down[:, :, e].T @ h)

    err = np.abs(o - o_ref)
    scale = np.abs(o_ref).max()
    print(f"D {D} F {F} E {E} K {K} T {T}: max |ref - c++| = {err.max():.3e}, max |o| = {scale:.3e}, "
          f"rel {err.max() / max(scale, 1e-30):.2e} {'PASS' if err.max() <= 1e-6 * max(scale, 1.0) else 'FAIL'}")
    _ = wts


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/moe.bin")
