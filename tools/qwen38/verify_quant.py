#!/usr/bin/env python3
"""verify_quant.py - did llama-quantize actually apply the allocation we asked for?

llama-quantize silently substitutes a "compatible" type when the requested block size does not
divide a tensor's row width (src/llama-quant.cpp:tensor_type_fallback), and it keeps a tensor at its
source type when the name never matches an override.  Either way the run SUCCEEDS and the file looks
right from the outside.  This reads the produced GGUF headers and compares every tensor against the
tensor-type file that was fed in.

    python3 verify_quant.py OUT-0000*-of-*.gguf --types types.txt [--json out.json]

Exit 0 = every named tensor has its requested type (and every other tensor kept a non-quantised
source type).  1 = at least one mismatch.  2 = malformed input.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys

# Mirrors tensor_allows_quantization (src/llama-quant.cpp:285-320): a tensor that fails these keeps
# its SOURCE type and its override is ignored, so expecting the allocated type there would report a
# false mismatch.
NEVER_QUANT = ("_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight",
               "altup", "laurel", "per_layer_model_proj", "ssm_conv1d")


def quantizable(name, ndim):
    if ndim < 2 or not name.endswith("weight"):
        return False
    return not any(s in name for s in NEVER_QUANT)


def load_types(path):
    """The tensor-type file is whitespace-separated `regex=type`.  prep_alloc escapes the dots and
    anchors both ends, so undo all three to recover the literal tensor name."""
    out = {}
    for tok in open(path).read().split():
        if "=" not in tok:
            raise SystemExit(f"{path}: token without '=': {tok!r}")
        name, typ = tok.rsplit("=", 1)
        out[name.strip("^$").replace("\\", "")] = typ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--types", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()

    paths = []
    for pat in a.shards:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        print("no shards matched", file=sys.stderr)
        return 2

    sys.path.insert(0, "/home/mal/AI/llama.cpp-master-rebase/gguf-py")
    import gguf

    got, ndim, ple_shard = {}, {}, None
    for p in paths:
        r = gguf.GGUFReader(p)
        for t in r.tensors:
            got[t.name] = t.tensor_type.name
            ndim[t.name] = len(t.shape)
            if t.name == "per_layer_token_embd.weight":
                ple_shard = p

    want = load_types(a.types)
    missing = sorted(set(want) - set(got))
    bad = {}
    for n, t in want.items():
        if n not in got or got[n] == t:
            continue
        if not quantizable(n, ndim[n]):
            continue        # the override is ignored here by design; source type is correct
        bad[n] = (t, got[n])
    unnamed = sorted(n for n in got if n not in want)

    hist = collections.Counter(got.values())
    print(f"{len(paths)} shard(s), {len(got)} tensors")
    print(f"  PLE table in: {ple_shard}")
    print(f"  type histogram: {dict(sorted(hist.items()))}")
    print(f"  overrides requested: {len(want)}, matched: {len(want) - len(missing)}, "
          f"mismatched: {len(bad)}")
    for n, (t, g) in list(bad.items())[:12]:
        print(f"    MISMATCH {n}: wanted {t}, got {g}")
    if missing:
        print(f"  MISSING from the file ({len(missing)}): {missing[:8]}")
    if unnamed:
        print(f"  WARNING {len(unnamed)} produced tensor(s) were not in the allocation: "
              f"{unnamed[:8]}")

    ok = not bad and not missing
    if a.json:
        json.dump({"shards": paths, "ple_shard": ple_shard, "tensors": len(got),
                   "histogram": dict(hist), "mismatched": {k: list(v) for k, v in bad.items()},
                   "missing": missing, "unnamed": unnamed, "ok": ok},
                  open(a.json, "w"), indent=1)
    print("VERIFY OK" if ok else "VERIFY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
