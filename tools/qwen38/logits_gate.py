#!/usr/bin/env python3
"""logits_gate.py - the honest quality gate: does the quantised file still predict what the parent predicts?

THE PARENT IS THE EDITED Q8, not a pristine model. The abliteration was applied in place, so the Q8
on disk today IS the parent of the XQ file. Comparing against it measures exactly one thing -
quantisation damage - which is the only thing the allocator can be wrong about.

WHY NOT tools/v41/logits_kl.py: that tool assumes a HEADERLESS dump (ds41_generate's format).
`strata generate --dump-logits` writes a 2 x int32 header {n_vocab, n_rows} first
(src/program/generate.cpp:6105). Feeding a strata dump to logits_kl.py reshapes the header bytes as
the first row's logits - every row shifts by 8 bytes and every number it prints is wrong while
looking authoritative. This tool parses the header and REFUSES a file whose size disagrees with it,
so a truncated dump fails loudly instead of being reinterpreted.

    strata generate --model M.gguf --tokens 1,2,3 --max-new 1 --dump-logits out.f32
    python3 logits_gate.py parent.f32 quant.f32 [--ids prompt.i32] [--json out.json]
                          [--min-top1 0.90] [--max-kl 0.05]

Exit 0 = ran (and passed, if thresholds were given).  1 = gate failed.  2 = malformed input.
With no thresholds it REPORTS ONLY: there is no calibrated pass mark yet, and inventing one would
be the same mistake as judging edit survival by a norm ratio.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

HEADER = 8  # two int32: n_vocab, n_rows


class DumpError(Exception):
    """The file is not the dump its header claims to be."""


def load(path: str, vocab: int | None = None) -> tuple[int, int, int]:
    """Return (n_vocab, n_rows, data_offset), verified against the file's real size.

    vocab is only for headerless dumps (ds41/llama.cpp). Passing it for a strata dump is an
    error on purpose: quietly reading a strata file as headerless is the failure this guards.
    """
    size = os.path.getsize(path)
    if vocab is not None:
        if size % (4 * vocab) != 0:
            raise DumpError(f"{path}: {size} bytes is not a whole number of {vocab}-wide f32 rows")
        return vocab, size // (4 * vocab), 0
    if size < HEADER:
        raise DumpError(f"{path}: {size} bytes, shorter than the {HEADER}-byte strata header")
    with open(path, "rb") as f:
        raw = f.read(HEADER)
    n_vocab, n_rows = struct.unpack("<2i", raw)
    if n_vocab <= 0 or n_rows <= 0:
        raise DumpError(f"{path}: header says n_vocab={n_vocab} n_rows={n_rows}; not a strata dump?"
                        " (pass --vocab for a headerless ds41/llama.cpp dump)")
    want = HEADER + n_vocab * n_rows * 4
    if want != size:
        raise DumpError(f"{path}: header claims {n_vocab} x {n_rows} = {want} bytes, file is {size}"
                        f" ({'truncated' if size < want else 'trailing data'}; a dump is written in one"
                        " pass, so this file is not a completed run)")
    return n_vocab, n_rows, HEADER


def log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def compare(a_path: str, b_path: str, vocab: int | None = None, ids_path: str | None = None,
            chunk: int = 64) -> dict:
    va, ra, oa = load(a_path, vocab)
    vb, rb, ob = load(b_path, vocab)
    if va != vb:
        raise DumpError(f"vocab mismatch: {va} vs {vb}")
    n = min(ra, rb)
    if ra != rb:
        # a different --max-new or prompt length; comparing the common prefix is still useful but
        # the caller must know the two runs were not identical
        print(f"warning: row counts differ ({ra} vs {rb}); comparing the first {n}", file=sys.stderr)
    A = np.memmap(a_path, dtype=np.float32, mode="r", offset=oa).reshape(ra, va)
    B = np.memmap(b_path, dtype=np.float32, mode="r", offset=ob).reshape(rb, vb)
    ids = np.fromfile(ids_path, dtype=np.int32) if ids_path else None
    if ids is not None and len(ids) < n + 1:
        raise DumpError(f"ids file has {len(ids)} tokens; row t predicts ids[t+1], need >= {n + 1}")

    kls, top1, dmax, nll_a, nll_b = [], 0, 0.0, [], []
    worst, worst_kl = 0, -1.0
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        la = log_softmax(np.asarray(A[r0:r1], dtype=np.float64))
        lb = log_softmax(np.asarray(B[r0:r1], dtype=np.float64))
        row_kl = (np.exp(la) * (la - lb)).sum(axis=1)
        kls.append(row_kl)
        i = int(np.argmax(row_kl))
        if row_kl[i] > worst_kl:
            worst_kl, worst = float(row_kl[i]), r0 + i
        top1 += int((la.argmax(1) == lb.argmax(1)).sum())
        dmax = max(dmax, float(np.abs(A[r0:r1] - B[r0:r1]).max()))
        if ids is not None:
            m = min(r1, n)
            for t in range(r0, m):
                if t + 1 < len(ids):
                    nll_a.append(-la[t - r0, ids[t + 1]])
                    nll_b.append(-lb[t - r0, ids[t + 1]])
    kl = np.concatenate(kls) if kls else np.zeros(0)
    out = {"rows": n, "vocab": va,
           "kl_mean": float(kl.mean()) if n else 0.0,
           "kl_p99": float(np.quantile(kl, 0.99)) if n else 0.0,
           "kl_max": float(kl.max()) if n else 0.0,
           "kl_worst_row": worst,
           "top1_agreement": top1 / n if n else 1.0,
           "max_abs_logit_diff": dmax}
    if ids is not None and nll_a:
        out["ppl_a"] = float(np.exp(np.mean(nll_a)))
        out["ppl_b"] = float(np.exp(np.mean(nll_b)))
        out["nll_tokens"] = len(nll_a)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="parent dump (edited Q8)")
    ap.add_argument("b", help="candidate dump (quantised file)")
    ap.add_argument("--vocab", type=int, default=None,
                    help="headerless dump width (ds41/llama.cpp format). Omit for strata dumps.")
    ap.add_argument("--ids", default=None, help="int32 token ids for the prompt+stream, for ppl")
    ap.add_argument("--json", default=None, help="write the metrics here")
    ap.add_argument("--min-top1", type=float, default=None)
    ap.add_argument("--max-kl", type=float, default=None)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args(argv)
    try:
        m = compare(a.a, a.b, a.vocab, a.ids, a.chunk)
    except DumpError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"{m['rows']} rows, vocab {m['vocab']}: KL mean {m['kl_mean']:.3e}  p99 {m['kl_p99']:.3e}  "
          f"max {m['kl_max']:.3e} (row {m['kl_worst_row']});  top-1 agreement {m['top1_agreement']:.4f};  "
          f"max |dlogit| {m['max_abs_logit_diff']:.4f}")
    if "ppl_a" in m:
        print(f"ppl parent {m['ppl_a']:.4f}  candidate {m['ppl_b']:.4f}  ({m['nll_tokens']} tokens)")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1)
    failed = []
    if a.min_top1 is not None and m["top1_agreement"] < a.min_top1:
        failed.append(f"top-1 {m['top1_agreement']:.4f} < {a.min_top1}")
    if a.max_kl is not None and m["kl_mean"] > a.max_kl:
        failed.append(f"KL mean {m['kl_mean']:.3e} > {a.max_kl:g}")
    if a.min_top1 is None and a.max_kl is None:
        print("REPORT ONLY (no thresholds given; nothing was gated)")
        return 0
    print("GATE", "FAIL" if failed else "PASS", ("(" + "; ".join(failed) + ")") if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
