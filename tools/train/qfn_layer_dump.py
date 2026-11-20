#!/usr/bin/env python3
"""Turn a strata `--dump-layers` file into (layer input, layer output) training pairs for the QFN trainer.

usage: qfn_layer_dump.py DUMP.bin --n-layers 48 --hc 4 --n-embd 2560 [--position P] [--layer L]
                           [--out-prefix PREFIX] [--summary]

Layout, from src/core/session.cpp:662-670 and 757-763 (verified in the source, not inferred):
    per POSITION, one record of (n_layers + 1) * hc * n_embd f32:
        slot 0        = the input residual (`s.R` when session_loop is entered = the embedding
                        broadcast to every hc stream - the reference's hc_init node)
        slot l+1      = the residual R after layer l        (l = 0 .. n_layers-1)
    each slot is hc * n_embd floats; `s.R` is stored stream-major (hc rows of n_embd).

`--out-prefix P` writes P.layerL.in.f32 and P.layerL.out.f32 (D * hc floats each) for one position and layer,
which is the shape qfn_train.cpp reads for `xa` / the target.  `--summary` prints per-slot RMS so a dump can be
sanity-checked before it is trusted (slot 0 should look like embeddings: small and roughly the same scale at every
position; the last slot should look like the final residual: much larger than the embedding).

NOTE on what this buys: matching slot l+1 from slot l is reconstruction (the layer's own output), not a preference
fine-tune.  It is the honest first target on real activations - with it, a failure cannot hide behind a synthetic
target, and a pass shows the graph reproduces the served layer on the model's own data.
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--hc", type=int, required=True)
    ap.add_argument("--n-embd", type=int, required=True)
    ap.add_argument("--position", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out-prefix")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    slot = a.hc * a.n_embd
    rec = (a.n_layers + 1) * slot
    size = np.memmap(a.dump, dtype=np.float32, mode="r").size
    if size % rec:
        print(f"ERROR: {size} floats is not a whole number of records ({rec} floats each) - wrong geometry?")
        return 1
    n_pos = size // rec
    print(f"{a.dump}: {n_pos} positions x {a.n_layers + 1} slots x {slot} floats "
          f"(hc {a.hc} x n_embd {a.n_embd}, {size * 4 / 1e6:.1f} MB)")

    m = np.memmap(a.dump, dtype=np.float32, mode="r", shape=(n_pos, a.n_layers + 1, a.hc, a.n_embd))
    if a.summary:
        print(f"{'pos':>4} {'slot0 (input)':>13} {'slot1':>10} {'slot-last':>10}   rms per slot")
        for p in range(min(n_pos, 8)):
            rms = np.sqrt((m[p] ** 2).mean(axis=(1, 2)))
            print(f"{p:>4} {rms[0]:>13.5f} {rms[1]:>10.5f} {rms[-1]:>10.5f}   "
                  f"min {rms.min():.4f} max {rms.max():.4f}")
        rms = np.sqrt((m ** 2).mean(axis=(0, 2, 3)))
        print(f"\nper-slot RMS over all positions: first 5 {np.round(rms[:5], 5)} ... last 3 {np.round(rms[-3:], 5)}")
        growth = rms[-1] / max(rms[0], 1e-12)
        print(f"final/input RMS ratio: {growth:.2f} (a served model grows the residual a lot; ~1 means the dump is wrong)")

    if a.out_prefix:
        if not (0 <= a.position < n_pos):
            print(f"ERROR: position {a.position} out of range (0..{n_pos - 1})")
            return 1
        if not (0 <= a.layer < a.n_layers):
            print(f"ERROR: layer {a.layer} out of range (0..{a.n_layers - 1})")
            return 1
        xin = np.ascontiguousarray(m[a.position, a.layer])       # slot `layer` is the INPUT to that layer
        xout = np.ascontiguousarray(m[a.position, a.layer + 1])  # slot layer+1 is its OUTPUT
        np.asarray(xin, dtype=np.float32).tofile(f"{a.out_prefix}.layer{a.layer}.in.f32")
        np.asarray(xout, dtype=np.float32).tofile(f"{a.out_prefix}.layer{a.layer}.out.f32")
        print(f"wrote {a.out_prefix}.layer{a.layer}.in.f32 / .out.f32 "
              f"({xin.size} floats each; in rms {np.sqrt((xin**2).mean()):.5f}, out rms {np.sqrt((xout**2).mean()):.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
