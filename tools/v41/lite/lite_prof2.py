import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
torch.set_num_threads(int(sys.argv[1]))
import v41_gptq as G
torch.set_default_dtype(torch.float32)
W = torch.randn(4608, 5120) * 0.02; d = torch.rand(5120) + 0.1
t = time.time(); ref = G.search_scales(W, d, "q2_0"); print("orig", round(time.time() - t, 2))
for rb in (16, 32, 64, 128, 256):
    t = time.time(); s = torch.cat([G.search_scales(W[i:i + rb], d, "q2_0") for i in range(0, W.shape[0], rb)])
    print(rb, round(time.time() - t, 2), "same" if torch.equal(s, ref) else f"DIFF {(s != ref).float().mean():.2e}")
