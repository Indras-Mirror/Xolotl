#!/usr/bin/env python3
"""Compare a real expert's deferred worker teacher with DeepSeek Expert.forward (CPU, exact bf16)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np
import torch
import lite_calib as LC


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", help="scratch directory; defaults to a new retained temporary directory")
    a = ap.parse_args()
    torch.set_num_threads(4)
    C = LC.setup_calib()
    from safetensors import safe_open
    wm = C.weight_map()
    L, ex = 3, 7
    pre = f"layers.{L}.ffn.experts.{ex}."
    with safe_open(C.shard_file(wm[pre + "w1.weight"]), "pt") as f:
        w = {k: f.get_tensor(pre + k) for k in ("w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale")}
    torch.set_default_dtype(torch.bfloat16)
    E = C.M.Expert(5120, 2304, dtype=torch.float4_e2m1fn_x2, swiglu_limit=10.0)
    for n in ("w1", "w3", "w2"):
        getattr(E, n).weight.data.view(torch.uint8).copy_(w[n + ".weight"].view(torch.uint8))
        getattr(E, n).scale.data.view(torch.uint8).copy_(w[n + ".scale"].view(torch.uint8))
    g = torch.Generator().manual_seed(0)
    xe = (torch.randn(700, 5120, generator=g) * 0.5).to(torch.bfloat16)
    ge = torch.rand(700, 1, generator=g, dtype=torch.float32)
    fit = torch.rand(700, generator=g, dtype=torch.float32) > 0.25
    with torch.inference_mode():
        ref = E(xe, ge)
    torch.set_default_dtype(torch.float32)
    out = Path(a.out_dir) if a.out_dir else Path(tempfile.mkdtemp(prefix="lite-selftest-"))
    (out / "q2").mkdir(parents=True, exist_ok=True)
    with (out / "q2/L03.bin").open("xb") as f:
        f.truncate(384 * LC.EXB)
    t = {"L": L, "ex": ex, "lim": 10.0, "min_fit": 32, "xe": xe.view(torch.int16).numpy(), "ge": ge.numpy(), "fit": fit.numpy()}
    for n in ("w1", "w3", "w2"):
        t[n] = w[n + ".weight"].view(torch.uint8).numpy()
        t["s" + n[1]] = w[n + ".scale"].view(torch.uint8).numpy()
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"), initializer=LC._winit,
                             initargs=(4, str(out), LC.q3k_offsets())) as pool:
        _, _, st, y = pool.submit(LC._work, t).result()
    y = torch.from_numpy(y).view(torch.bfloat16)
    assert torch.equal(y, ref), f"teacher differs: max abs {float((y.float()-ref.float()).abs().max())}"
    assert all(np.isfinite(st[k]) for k in ("e33", "e23", "e32", "e22", "y_energy")), st
    print("PASS: deferred worker teacher equals Expert.forward bitwise; finite held-out errors; scratch", out)


if __name__ == "__main__":
    main()
