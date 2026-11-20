#!/usr/bin/env python3
"""Gate: serve-mode captures (harness_capture_serve.py) against cold one-shot captures (harness_capture.py).

usage: harness_capture_compare.py <cold-dir> <serve-dir>

For every window captured in both: x44 (and x45/x46/x47/final when both kept them) max |diff| and relative error, and
layer 47's selected cells per window query (exact match fraction). Cold reads are the reference; a serve read resumes
from the root checkpoint at the prefix end, so any difference is the checkpoint path (or chunk boundaries).
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROW = 4 * 2560


def f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def sel(path: Path, P: int, n: int):
    cells = {}
    b = path.read_bytes()
    o = 0
    while o + 16 <= len(b):
        layer, p0, T, cap = struct.unpack_from("4i", b, o)
        o += 16
        ids = np.frombuffer(b, dtype=np.int32, count=T * cap, offset=o).reshape(T, cap)
        o += T * cap * 4
        for t in range(T):
            wp = p0 + t - P
            if 0 <= wp < n:
                cells[wp] = ids[t]
    return cells


def main():
    cold, serve = Path(sys.argv[1]), Path(sys.argv[2])
    P = 20135
    worst = 0.0
    for sd in sorted(serve.glob("w*")):
        cd = cold / sd.name
        if not (cd / "meta.json").exists() or not (sd / "meta.json").exists():
            continue
        m = json.loads((sd / "meta.json").read_text())
        n = m["n"]
        parts = []
        for name in ("x44", "x45", "x46", "x47", "final"):
            a, b = cd / f"{name}.bf16", sd / f"{name}.bf16"
            if not (a.exists() and b.exists()):
                continue
            x, y = f32(np.fromfile(a, dtype=np.uint16)), f32(np.fromfile(b, dtype=np.uint16))
            if x.size != y.size:
                parts.append(f"{name} SIZE {x.size} vs {y.size}")
                continue
            d = float(np.abs(x - y).max())
            rel = float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30))
            worst = max(worst, rel)
            parts.append(f"{name} max {d:.2e} rel {rel:.2e}")
        sc, ss = sel(cd / "qsa.bin", P, n), sel(sd / "qsa.bin", P, n)
        common = sorted(set(sc) & set(ss))
        same = sum(1 for t in common if np.array_equal(np.sort(sc[t]), np.sort(ss[t])))
        parts.append(f"qsa {same}/{len(common)} queries identical ({len(sc)} cold, {len(ss)} serve)")
        print(f"{sd.name} (serve reads from {m.get('read_starts')}): " + " | ".join(parts))
    print(f"worst relative error {worst:.2e}")


if __name__ == "__main__":
    main()
