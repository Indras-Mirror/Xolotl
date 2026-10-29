#!/usr/bin/env python3
"""prep_alloc.py - turn a published per-tensor allocation into a llama-quantize tensor-type file,
and predict the output size BEFORE writing a byte.

WHY THIS EXISTS
  `llama-quantize --tensor-type-file` reads whitespace-separated `name=type` tokens and every token
  must contain '=' (tools/quantize/quantize.cpp:parse_tensor_type_file).  The published GSQ-RCO
  allocation files start with '#' comment lines full of spaces, so feeding one directly aborts with
  "malformed tensor type".  The comments must be stripped first.

  Overrides are compiled as std::regex and matched with std::regex_search
  (src/llama-quant.cpp:694,:717), i.e. UNANCHORED and first-match-wins in file order.  Raw names
  would treat every '.' as "any char", and 'token_embd.weight' is a substring of
  'per_layer_token_embd.weight' - which only worked by luck of ordering.  Every pattern emitted here
  is `^name$` anchored, and every pattern is compiled here so a regex error fails in seconds at this
  stage instead of after the pack, inside C++ (there is no try/catch around the compile).

EVERY TENSOR IS EMITTED, INCLUDING THE ONES THAT "STAY AS THEY ARE"
  It is tempting to omit the F32/BF16 tensors and let them default.  That is wrong: under
  `Q8_0` ftype the default target is Q8_0, and `tensor_allows_quantization` (src/llama-quant.cpp:
  285-320) only excludes 1-D tensors, `*_norm.weight`, `ffn_gate_inp.weight` and `ssm_conv1d`.
  The hyper-connection projections (`hc_*`) are 2-D and DO allow quantisation, so omitting them
  would silently quantise BF16 down to Q8_0 - destroying precision the engine and the published
  allocation both expect.  Naming them explicitly makes the target equal the source and the tensor
  is copied byte for byte.

  Conversely a name that cannot be quantised has its override IGNORED and keeps its source type
  (`llama_tensor_get_type` returns `tensor->type` first).  Both the estimate and verify_quant.py
  model that rule, so the prediction matches what llama-quantize actually produces.

    python3 prep_alloc.py ALLOC.txt --out types.txt --estimate --shard-dir DIR
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

import numpy as np

# Mirrors tensor_allows_quantization (src/llama-quant.cpp:285-320).  A tensor outside these rules
# keeps its source type no matter what the allocation says.
NEVER_QUANT = ("_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight",
               "altup", "laurel", "per_layer_model_proj", "ssm_conv1d")


def quantizable(name: str, ndim: int, output_quantised: bool = True) -> bool:
    if ndim < 2:
        return False
    if not name.endswith("weight"):
        return False
    if any(s in name for s in NEVER_QUANT):
        return False
    if name == "output.weight" and not output_quantised:
        return False
    return True


def parse_alloc(path):
    """-> ordered dict name -> type, skipping '#' comments and blank lines."""
    out = collections.OrderedDict()
    for lineno, line in enumerate(open(path), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"{path}:{lineno}: not name=type: {line!r}")
        name, typ = line.rsplit("=", 1)
        out[name.strip()] = typ.strip()
    return out


def pattern(name):
    """The std::regex llama-quantize will compile: exact, escaped, anchored."""
    return "^" + re.escape(name) + "$"


def quant_sizes():
    sys.path.insert(0, "/home/mal/AI/llama.cpp-master-rebase/gguf-py")
    import gguf
    return {k.name: v for k, v in gguf.GGML_QUANT_SIZES.items()}


def read_shapes(shard_dir):
    """name -> (element count, row width ne0, source type name), from the readable shards.

    ne0 matters more than the element count: ggml quantises a row at a time, so the block size must
    divide ne0, not merely the total.  ffn_down_exps is [640, 2560, 512] and 640 % 256 != 0, which is
    exactly why it cannot hold IQ2_*/IQ3_* at all while ffn_gate_exps ([2560, ...]) can.  Checking
    only the product would let a wrong allocation through and llama-quantize would silently fall
    back to a compatible type (src/llama-quant.cpp:tensor_type_fallback).
    """
    sys.path.insert(0, "/home/mal/AI/llama.cpp-master-rebase/gguf-py")
    import gguf
    shapes = {}
    for p in sorted(pathlib.Path(shard_dir).glob("*.gguf")):
        try:
            r = gguf.GGUFReader(str(p))
        except Exception as exc:            # a shard still downloading has no readable header
            print(f"  (skipping unreadable shard {p.name}: {type(exc).__name__})", file=sys.stderr)
            continue
        for t in r.tensors:
            dims = [int(x) for x in t.shape]
            src = getattr(t.tensor_type, "name", str(t.tensor_type))
            shapes[t.name] = (int(np.prod(dims)), dims[0] if dims else 0, src, len(dims))
    return shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alloc")
    ap.add_argument("--out", help="write the llama-quantize tensor-type file here")
    ap.add_argument("--estimate", action="store_true", help="print the predicted output size")
    ap.add_argument("--shard-dir", help="GGUF shard dir, for exact shapes and source types")
    ap.add_argument("--assume-elems", action="append", default=[],
                    help="name=count for a tensor in an unreadable shard")
    ap.add_argument("--leave-output", action="store_true",
                    help="assume llama-quantize was given --leave-output-tensor")
    ap.add_argument("--max-gb", type=float, default=0.0, help="fail if the estimate exceeds this")
    a = ap.parse_args()

    alloc = parse_alloc(a.alloc)
    sizes = quant_sizes()

    bad = sorted({t for t in alloc.values()} - set(sizes))
    if bad:
        raise SystemExit(f"unknown ggml types in allocation: {bad}")

    # Every pattern is compiled here: llama-quant.cpp:193-199 does not catch std::regex_error, so a
    # bad pattern would TERMINATE llama-quantize at stage 6 instead of failing here at stage 4.
    lines = []
    for name, typ in alloc.items():
        pat = pattern(name)
        try:
            re.compile(pat)
        except re.error as exc:
            raise SystemExit(f"{name}: pattern {pat!r} does not compile: {exc}")
        if "#" in pat:
            raise SystemExit(f"{name}: pattern contains '#' - llama-quantize would abort")
        lines.append(f"{pat}={typ}")

    if a.out:
        pathlib.Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"{a.out}: {len(lines)} overrides "
              f"({sum(1 for t in alloc.values() if t != 'Q8_0')} change the type)")

    if not a.estimate:
        return 0

    shapes = read_shapes(a.shard_dir) if a.shard_dir else {}
    assumed = {s.rsplit("=", 1)[0]: int(s.rsplit("=", 1)[1]) for s in a.assume_elems}
    total = 0
    missing, fallbacks = [], []
    hist = collections.Counter()
    for name, typ in alloc.items():
        got = shapes.get(name)
        if got is None:
            if name in assumed:
                nelem, ne0, src, ndim = assumed[name], None, typ, 2
            else:
                missing.append(name)
                continue
        else:
            nelem, ne0, src, ndim = got

        # llama-quantize ignores the override on a tensor it will not quantise
        eff = typ if quantizable(name, ndim, not a.leave_output) else src
        blk, tsize = sizes[eff]
        if nelem % blk:
            raise SystemExit(f"{name}: {nelem} elements is not a multiple of {eff}'s block {blk}")
        if ne0 is not None and ne0 % blk:
            # llama-quantize would silently substitute a compatible type here
            fallbacks.append((name, eff, ne0, blk))
            eff = src
            blk, tsize = sizes[src]
            if nelem % blk:
                raise SystemExit(f"{name}: cannot hold {typ} (ne0={ne0}), and its source type "
                                 f"{src} does not divide {nelem} either")
        total += nelem // blk * tsize
        hist[eff] += 1

    print(f"\npredicted size {total/1e9:.2f} GB  ({total/2**30:.2f} GiB)  "
          f"from {len(alloc) - len(missing)} tensors")
    print(f"  type histogram: {dict(sorted(hist.items()))}")
    if fallbacks:
        print(f"  {len(fallbacks)} tensor(s) CANNOT hold their allocated type (row width not "
              f"divisible); llama-quantize will fall back and they keep the source type:")
        for n, t, ne0, blk in fallbacks[:10]:
            print(f"    {n}: {t} needs ne0 %% {blk} == 0, ne0={ne0}")
        if len(fallbacks) > 10:
            print(f"    ... and {len(fallbacks) - 10} more")
    if missing:
        print(f"  MISSING SHAPE ({len(missing)}): {missing}")
        print("  -> estimate is LOW by those tensors; re-run once the shard is readable")
        return 3
    if a.max_gb and total / 1e9 > a.max_gb:
        print(f"  OVER CAP: {total/1e9:.2f} GB > {a.max_gb} GB")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
