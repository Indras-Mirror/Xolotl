#!/usr/bin/env python3
"""lite_prof.py - time the stages of one lite_calib expert job on random rows (threads = argv[1], rows = argv[2])."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import torch
torch.set_num_threads(int(sys.argv[1]))
import v41_calib as C
import v41_quant as Q
import v41_formats as F
import v41_gptq as G
torch.set_default_dtype(torch.float32)
n = int(sys.argv[2])
D, FF = 5120, 2304
g = torch.Generator().manual_seed(0)
W1, W3 = torch.randn(FF, D, generator=g) * 0.02, torch.randn(FF, D, generator=g) * 0.02
W2 = torch.randn(D, FF, generator=g) * 0.02
xe = (torch.randn(n, D, generator=g)).to(torch.bfloat16)
ge = torch.rand(n, 1, generator=g)
fit = torch.rand(n, generator=g) > 0.25
T = {}
t = time.time()
H1, H2 = C.act_hessians(xe[fit], W1, W3, ge[fit], 10.0); T["hess"] = time.time() - t
t = time.time(); d = torch.diag(H1).clamp_min(0) + 1e-8
S = [G.search_scales(W, d, "q2_0") for W in (W1, W3)]; T["scales_gu"] = time.time() - t
t = time.time(); Hinv, dead = G.hinv_chol(H1, 1.0); T["chol"] = time.time() - t
Wst = torch.cat([W1, W3]); Sst = torch.cat([G.expand_scales(s, "q2_0", D) for s in S])
lo = torch.full((Wst.shape[0], 1), -1.0); hi = torch.full((Wst.shape[0], 1), 2.0)
t = time.time(); Qg = G.gptq(Wst, Sst, lo, hi, Hinv, dead); T["gptq_gu"] = time.time() - t
t = time.time(); (g2,) = Q.quant([W2], H2, "q2_0", 1.0, True); T["quant_dn_all"] = time.time() - t
t = time.time(); b = F.encode_tier(Qg[:FF].numpy(), S[0].numpy(), "q2_0"); T["encode1"] = time.time() - t
raw = np.random.default_rng(0).integers(0, 255, FF * D // 256 * 110, dtype=np.uint8).tobytes()
t = time.time(); F.decode(F.Q3_K, raw, FF * D); T["q3dec1"] = time.time() - t
xv, gv = xe[~fit], ge[~fit]
t = time.time(); Y = C.expert_f(xv, W1, W3, W2, gv, 10.0); T["teacher"] = time.time() - t
t = time.time(); C.sq_err(xv, W1, W3, W2, gv, 10.0, Y); T["sqerr1"] = time.time() - t
print({k: round(v, 2) for k, v in T.items()})
