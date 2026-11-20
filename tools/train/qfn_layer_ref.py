#!/usr/bin/env python3
"""Numpy reference for test_qfn_layer / tools/train/qfn_layer_train.hpp (one QFN gated-delta-net layer).

usage: qfn_layer_ref.py dump.bin

Recomputes the layer from the dumped weights and inputs and compares against the C++ graph output.
"""
import sys

import numpy as np

import gdn_ref


def silu(x):
    return x / (1.0 + np.exp(-x))


def main(path):
    raw = open(path, "rb").read()
    D, hd, nk, nv, T = [int(v) for v in np.frombuffer(raw[:20], dtype=np.int32)]
    d_conv = 4
    n_qkv = 2 * hd * nk + hd * nv
    n_z = hd * nv
    off = 20

    def take(ne):
        nonlocal off
        n = int(np.prod(ne)) * 4
        a = np.frombuffer(raw[off:off + n], dtype=np.float32).reshape(ne[::-1]).transpose(
            *range(len(ne) - 1, -1, -1)).astype(np.float64)
        off += n
        return a

    cur = take((D, T))
    wqkv = take((D, n_qkv + n_z))
    convw = take((d_conv, n_qkv))
    wba = take((D, 2 * nv))
    ssm_a = take((nv,))
    dt = take((nv,))
    nrm = take((hd,))
    wout = take((hd * nv, D))
    _r = take((D, T))
    o_ref = take((D, T))

    qkvz = wqkv.T @ cur                                     # [n_qkv + n_z, T]
    qkv, zt = qkvz[:n_qkv], qkvz[n_qkv:]
    cin = np.zeros((d_conv - 1 + T, n_qkv))
    cin[d_conv - 1:] = qkv.T
    y = np.zeros((n_qkv, T))
    for k in range(d_conv):
        y += convw[k][:, None] * cin[k:k + T].T
    y = silu(y)

    q = y[:hd * nk].T.reshape(T, nk, hd).transpose(2, 1, 0)
    k_ = y[hd * nk:2 * hd * nk].T.reshape(T, nk, hd).transpose(2, 1, 0)
    v = y[2 * hd * nk:].T.reshape(T, nv, hd).transpose(2, 1, 0)

    def l2(t):
        return t / np.sqrt((t ** 2).mean(axis=0, keepdims=True) + 1e-6 / hd) * (1.0 / np.sqrt(hd))

    q = np.tile(l2(q), (1, nv // nk, 1))
    k_ = np.tile(l2(k_), (1, nv // nk, 1))

    ba = wba.T @ cur
    beta = 1.0 / (1.0 + np.exp(-ba[:nv]))
    gate = np.log1p(np.exp(ba[nv:] + dt[:, None])) * ssm_a[:, None]     # softplus * (-A_log.exp()) <= 0

    o = gdn_ref.gdn_forward(q, k_, v, gate[None], beta[None], hd, nv, T)

    z = zt.T.reshape(T, nv, hd).transpose(2, 1, 0)
    on = o / np.sqrt((o ** 2).mean(axis=0, keepdims=True) + 1e-6) * nrm[:, None, None]
    on = on * (1.0 / (1.0 + np.exp(-z)))
    out = wout.T @ on.transpose(2, 1, 0).reshape(T, nv * hd).T
    layer = cur + out

    err = np.abs(layer - o_ref)
    scale = np.abs(o_ref).max()
    print(f"D {D} hd {hd} nk {nk} nv {nv} T {T}: max |ref - c++| = {err.max():.3e}, max |o| = {scale:.3e}, "
          f"rel {err.max() / max(scale, 1e-30):.2e} {'PASS' if err.max() <= 1e-5 * max(scale, 1.0) else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qfnl.bin")
