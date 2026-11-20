#!/usr/bin/env python3
"""Numpy reference for test_gdn_layer (QFN layer core: conv -> silu -> split -> l2 norm -> head tile -> GDN).

usage: gdn_layer_ref.py dump.bin

Recomputes the composition from the same inputs and compares against the C++ output, so the test checks that the
graph *is* the model's layer core, not merely that its gradient is self-consistent.
"""
import sys

import numpy as np

import gdn_ref


def main(path):
    raw = open(path, "rb").read()
    S, nk, nv, T, d_conv = [int(v) for v in np.frombuffer(raw[:20], dtype=np.int32)]
    channels = 2 * S * nk + S * nv
    n_cin = d_conv - 1 + T
    off = 20
    sizes = [(n_cin, channels), (d_conv, channels), (nv, T), (nv, T), (S, nv, T), (S, nv, T)]
    arrs = []
    for ne in sizes:
        n = int(np.prod(ne)) * 4
        arrs.append(np.frombuffer(raw[off:off + n], dtype=np.float32).reshape(ne[::-1]).transpose(
            *range(len(ne) - 1, -1, -1)).astype(np.float64))
        off += n
    x, c, g, b, r, o_ref = arrs
    x = x.T                                                   # [channels, n_cin] (dim0 is the conv window axis)

    # ssm_conv: y[i, t] = sum_k sx[t + k, i] * c[k, i], sx has d_conv-1 history rows then the tokens
    y = np.zeros((channels, T))
    for k in range(d_conv):
        y += c[k][:, None] * x[:, k:k + T]
    y = y * (1.0 / (1.0 + np.exp(-y)))                      # silu
    q = y[:S * nk].T.reshape(T, nk, S).transpose(2, 1, 0)     # [S, nk, T]
    k_ = y[S * nk:2 * S * nk].T.reshape(T, nk, S).transpose(2, 1, 0)
    v = y[2 * S * nk:].T.reshape(T, nv, S).transpose(2, 1, 0)  # [S, nv, T]
    eps = 1e-6

    def l2(t):                                                # [S, H, T]
        return t / np.sqrt((t ** 2).mean(axis=0, keepdims=True) + eps / S) * (1.0 / np.sqrt(S))

    q = l2(q); k_ = l2(k_)
    q = np.tile(q, (1, nv // nk, 1))                          # v head j <- k head j % nk
    k_ = np.tile(k_, (1, nv // nk, 1))
    o = gdn_ref.gdn_forward(q, k_, v, g[None], b[None], S, nv, T)

    err = np.abs(o - o_ref)
    scale = np.abs(o_ref).max()
    print(f"S {S} nk {nk} nv {nv} T {T}: max |ref - c++| = {err.max():.3e}, max |o| = {scale:.3e}, "
          f"rel {err.max() / max(scale, 1e-30):.2e} {'PASS' if err.max() <= 1e-5 * max(scale, 1.0) else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/glayer.bin")
