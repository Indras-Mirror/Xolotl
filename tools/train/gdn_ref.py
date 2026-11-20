#!/usr/bin/env python3
"""Numpy reference for tools/train/gdn_train.hpp (chunked gated delta net, non-KDA path).

Reads the dump written by test_gdn_chunked (S,H,T + q,k,v,g,b,R,o in ggml layout), recomputes the
forward from the llama.cpp delta-net-base.cpp algorithm with the ggml op conventions
(tri keeps d0<d1 for LOWER; cumsum inclusive along d0; mul_mat(a,b) = a^T b; solve_tri(A,B) = A^-T B),
and compares against the C++ output.  usage: gdn_ref.py dump.bin
"""
import sys
import numpy as np

CS = 64


def gg(buf, ne):
    """ggml dense tensor (dim0 fastest) -> numpy array indexed [d0, d1, d2, d3]."""
    a = np.frombuffer(buf, dtype=np.float32, count=int(np.prod(ne))).reshape(ne[::-1])
    return np.ascontiguousarray(a.transpose(3, 2, 1, 0))


def tri(a, keep):
    """ggml tri on the first two dims: LOWER keeps d0 < d1, LOWER_DIAG keeps d0 <= d1."""
    jj, ii = np.indices(a.shape[:2])
    mask = (jj < ii) if keep == "lower" else (jj <= ii)
    out = a.copy()
    out[~mask] = 0.0
    return out


def solve_tri(a, b):
    """ggml_solve_tri(a, b, left, lower, unit) = b @ a^-1 (verified numerically against ops3.cpp):
    rows of the answer are solved along dim1, A is stored with its triangle at d0 < d1."""
    return np.linalg.solve(a.T.astype(np.float64), np.ascontiguousarray(b.T).astype(np.float64)).T


def gdn_forward(q, k, v, g, b, S, H, T):
    """Chunked gated delta net, zero initial state. q/k/v: [S, H, T], g/b: [1, H, T] -> o: [S, H, T]."""
    pad = (CS - T % CS) % CS
    Tp = T + pad
    n_chunks = Tp // CS
    ident = np.eye(CS)
    o = np.zeros((S, H, T), dtype=np.float64)

    for h in range(H):
        state = np.zeros((S, S))
        for c in range(n_chunks):
            sl = slice(c * CS, (c + 1) * CS)
            qc = np.zeros((S, CS)); kc = np.zeros((S, CS)); vc = np.zeros((S, CS))
            gc = np.zeros(CS); bc = np.zeros(CS)
            lo, hi = c * CS, min((c + 1) * CS, T)
            if lo < T:
                qc[:, :hi - lo] = q[:, h, lo:hi]
                kc[:, :hi - lo] = k[:, h, lo:hi]
                vc[:, :hi - lo] = v[:, h, lo:hi]
                gc[:hi - lo] = g[0, h, lo:hi]
                bc[:hi - lo] = b[0, h, lo:hi]
            qc = qc / np.sqrt(S)                               # gdn_chunked scales q by 1/sqrt(S_k)
            G = np.cumsum(gc)                                   # inclusive cumsum along tokens
            D = np.triu(np.exp(G[None, :] - G[:, None]), 0)     # D[j, i] = exp(G_i - G_j), j <= i
            kb = (kc.T @ (kc * bc)) * D                         # [j, i]
            kq = np.triu((kc.T @ qc) * D, 0)
            attn = np.triu(kb, 1)
            lhs = attn + ident
            A = solve_tri(lhs, -attn) + ident
            v_new_all = (vc * bc) @ A                           # [S, CS]
            g_exp = np.exp(G)[:, None]                          # [CS, 1]
            k_cd = (g_exp * (kc * bc).T).T @ A                  # [S, CS]
            q_g_exp = qc * np.exp(G)[None, :]
            kd_t = (kc * np.exp(G[-1] - G)[None, :]).T          # [CS, S]
            v_t = v_new_all.T                                   # [CS, S]
            v_t_new = v_t - k_cd.T @ state
            o_ch = v_t_new.T @ kq + state.T @ q_g_exp
            if c * CS < T:
                o[:, h, c * CS:min((c + 1) * CS, T)] = o_ch[:, :min(CS, T - c * CS)]
            state = state * np.exp(G[-1]) + kd_t.T @ v_t_new

    return o


def main(path):
    raw = open(path, "rb").read()
    S, H, T = np.frombuffer(raw[:12], dtype=np.int32)
    S, H, T = int(S), int(H), int(T)
    off = 12
    arrs = []
    for ne in [(S, H, T, 1)] * 3 + [(1, H, T, 1)] * 2 + [(S, H, T, 1)] * 2:
        n = int(np.prod(ne)) * 4
        arrs.append(gg(raw[off:off + n], ne))
        off += n
    q, k, v, g, b, r, o_ref = [a[..., 0] for a in arrs]   # drop the 4th (n_seqs) dim

    o = gdn_forward(q, k, v, g, b, S, H, T)
    pad = (CS - T % CS) % CS
    n_chunks = (T + pad) // CS

    err = np.abs(o - o_ref[:, :, :T])
    scale = np.abs(o_ref[:, :, :T]).max()
    print(f"S {S} H {H} T {T} (pad {pad}, {n_chunks} chunks): max |ref - c++| = {err.max():.3e}, "
          f"max |o| = {scale:.3e}, rel {err.max() / max(scale, 1e-30):.2e} "
          f"{'PASS' if err.max() <= 1e-4 * max(scale, 1.0) else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gdn.bin")
