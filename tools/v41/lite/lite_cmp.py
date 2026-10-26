#!/usr/bin/env python3
"""lite_cmp.py A.gguf B.gguf: byte compare of the tensors both files hold (dense half; uniform/mixed expert
tensors resolved per expert through the I32 type maps): reports which differ."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import v41_gguf as W
import v41_formats as F
a, b = sys.argv[1], sys.argv[2]
ka, ta, sa = W.read_gguf(a); kb, tb, sb = W.read_gguf(b)
A = {n: (s, t, o) for n, s, t, o in ta}; B = {n: (s, t, o) for n, s, t, o in tb}
fa, fb = open(a, "rb"), open(b, "rb")
def rd(f, start, o, n): f.seek(start + o); return f.read(n)
same = diff = 0
for n in sorted(set(A) & set(B)):
    s1, t1, o1 = A[n]; s2, t2, o2 = B[n]
    if "_exps" in n: continue
    nb = F.nbytes(t1, s1)
    if (s1, t1) != (s2, t2) or rd(fa, sa, o1, nb) != rd(fb, sb, o2, nb): diff += 1; print("DIFF", n, s1, t1, s2, t2)
    else: same += 1
print(f"dense: {same} equal, {diff} differ; only in A: {len(set(A)-set(B))}, only in B: {len(set(B)-set(A))}")
# experts: per (layer, role, expert) through type maps
def exp_bytes(T, f, start, L, role, e):
    key = "down" if role == "down" else "gu"
    m = T.get(f"blk.{L}.ffn_exps_{key}_type")
    if m is None:
        s, t, o = T[f"blk.{L}.ffn_{role}_exps.weight"]; slot = e
    else:
        tm = np.frombuffer(rd(f, start, m[2], 384 * 4), np.int32); t = int(tm[e]); slot = int((tm[:e] == t).sum())
        s, t, o = T[f"blk.{L}.ffn_{role}_exps.{F.TYPE_NAME[t].lower()}.weight"]
    per = F.nbytes(t, s[:2]); return t, rd(f, start, o + slot * per, per)
Ls = sorted({int(n.split(".")[1]) for n in B if n.startswith("blk.") and "_exps" in n})
eq = ne_ = q2 = 0
for L in Ls:
    for e in range(384):
        for role in ("gate", "up", "down"):
            t2, x2 = exp_bytes(B, fb, sb, L, role, e)
            if t2 == F.Q3_K:
                t1, x1 = exp_bytes(A, fa, sa, L, role, e)
                if x1 == x2: eq += 1
                else: ne_ += 1
            else: q2 += 1
print(f"experts layers {Ls[0]}-{Ls[-1]}: Q3_K equal to A {eq}, differ {ne_}; Q2_0 matrices {q2}")

fa.close(); fb.close()
sys.exit(1 if diff or ne_ else 0)
