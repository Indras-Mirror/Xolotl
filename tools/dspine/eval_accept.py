"""Offline acceptance on OUR 27B (IQ3_S Swift-1.5 RCO, greedy, thinking off): walks each teacher-forced sequence the way
the speculative loop would (anchor, draft a block, accept the leading matches against the engine's own greedy tokens,
move on).  python eval_accept.py dspine|dflash2 [block sizes]"""
import json, struct, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/mal/AI/Xolotl/third_party/llama.cpp/gguf-py")
sys.path.insert(0, "/home/mal/AI/Xolotl/tools/dflash2")
import gguf
from gguf.quants import dequantize

W = "/home/mal/v41-work/dspine-20261109"
GGUF = "/media/Crucial1TB/Xolotl/qwen3.8-27b/Swift-1.5-27B-GSQ-RCO/d0xin-unc-27b-IQ3_S-mtp-q6embd.gguf"
which = sys.argv[1]
blocks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [16, 8]
dev = "cuda"
import os
TAPS = [int(x) for x in os.environ.get("TAPL", "1,5,16,19,31,33,46,47,61").split(",")]
TF = os.environ.get("TAPF", "taps.bin")
SHIFT = int(os.environ.get("SHIFT", "0"))
USE_TOP1 = os.environ.get("TOP1") == "1"

r = gguf.GGUFReader(GGUF)
T = {t.name: t for t in r.tensors}
emb_raw, head_raw = np.asarray(T["token_embd.weight"].data), np.asarray(T["output.weight"].data)
QT = gguf.GGMLQuantizationType.Q6_K


def embed_rows(ids):
    ii = ids.detach().cpu().numpy()
    return torch.from_numpy(dequantize(emb_raw[ii], QT).astype(np.float32)).to(dev)


t0 = time.time()
head = torch.empty(head_raw.shape[0], 5120, dtype=torch.bfloat16, device=dev)
for a in range(0, head_raw.shape[0], 16384):
    head[a:a + 16384] = torch.from_numpy(dequantize(head_raw[a:a + 16384], QT).astype(np.float32)).to(dev).to(torch.bfloat16)
print("head dequantized in %.0f s" % (time.time() - t0), flush=True)

# tap records: int32 n0, rows, n_taps, n_embd; int32 tokens[rows]; float rows[rows][n_taps*n_embd]
recs = []
with open(W + "/" + TF, "rb") as f:
    while True:
        h = f.read(16)
        if len(h) < 16:
            break
        n0, n, nt, E = struct.unpack("<4i", h)
        tok = np.frombuffer(f.read(4 * n), np.int32)
        rows = np.frombuffer(f.read(4 * n * nt * E), np.float32).reshape(n, nt, E)
        recs.append((n0, tok, rows))
seqs = json.load(open(W + "/seqs.json"))
assert len(recs) == len(seqs), (len(recs), len(seqs))

if which == "dspine":
    from dspine_ref import DSpine
    d = DSpine("/media/mal/NVME1TB/Xolotl/qwen3.8-27b/tencent-Qwen3.8-27B-DSpine-b16", embed_rows, head, dev)
else:
    torch.set_default_device(dev)
    import dflash2_ref as R
    SRC = "/media/mal/NVME1TB/Xolotl/qwen3.8-27b/z-lab-Qwen3.8-27B-DFlash2"

    class Rows:
        def __getitem__(self, ids):
            return embed_rows(torch.as_tensor(ids))

    class HeadF:
        def __getitem__(self, ids):
            return head[ids].float()

    d = R.Drafter(SRC + "/model.safetensors", SRC + "/config.json", Rows(), HeadF())
    d.w = {k: v.to(dev) for k, v in d.w.items()}
    d.inv_freq = d.inv_freq.to(dev)
    d.head_bf16, d.fast_head = head, True
cols = [TAPS.index(l + SHIFT) for l in d.taps]
print(which, "taps", d.taps, "columns", cols, flush=True)

res = {}
for (n0, tok, rows), sq in zip(recs, seqs):
    seq = sq["prompt"] + sq["out"]
    assert n0 == 0 and list(tok) == seq, sq["name"]
    feats = torch.from_numpy(rows[:, cols, :].reshape(len(seq), -1).copy()).to(dev)
    d.set_context(feats)
    res[sq["name"]] = {}
    for b in blocks:
        s, cyc = len(sq["prompt"]), []
        cond_ok, cond_n = np.zeros(b - 1), np.zeros(b - 1)
        t1 = time.time()
        with torch.no_grad():
            while s + 1 < len(seq):
                path = d.draft(s, seq[s], b)[1 if USE_TOP1 else 0]
                truth = seq[s + 1:s + b]
                a = 0
                while a < len(truth) and path[a] == truth[a]:
                    a += 1
                for i in range(min(a + 1, len(truth))):
                    cond_n[i] += 1
                    cond_ok[i] += (i < a)
                cyc.append(a + 1)
                s += a + 1
        tau = float(np.mean(cyc))
        res[sq["name"]]["b%d" % b] = {"rounds": len(cyc), "tau": tau, "cond": (cond_ok / np.maximum(cond_n, 1)).round(3).tolist()}
        print("%-9s block %2d: %3d rounds, %.2f tokens per round (%.0f s)" % (sq["name"], b, len(cyc), tau, time.time() - t1), flush=True)
for b in blocks:
    k = "b%d" % b
    tot_tok = sum(len(sq["out"]) - 1 for sq in seqs)
    tot_rounds = sum(res[n][k]["rounds"] for n in res)
    print("%s block %d: mean of prompts %.2f, pooled %.2f tokens per round" %
          (which, b, np.mean([res[n][k]["tau"] for n in res]), tot_tok / tot_rounds), flush=True)
json.dump(res, open(W + "/accept-%s-s%d-t%d.json" % (which, SHIFT, USE_TOP1), "w"), indent=1)
