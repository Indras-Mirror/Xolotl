#!/usr/bin/env python3
"""Turn a v41_alloc.py map into a llama-quantize --tensor-type-file.

`llama-quantize` stores ONE ggml type per tensor, and a qwen4exp layer's routed experts are a
single 3D tensor per projection (`blk.N.ffn_gate_exps.weight` holds all 512 experts).  So the
per-expert decisions XQ makes for DeepSeek cannot be expressed in a plain GGUF: they have to be
collapsed to one tier per (layer, projection).  This script does that collapse explicitly rather
than silently picking the first expert.

Collapse rule: the tier covering the most expert-bytes in that (layer, part) wins; ties go to the
HIGHER bits-per-weight.  "Most bytes" is the same quantity the allocator optimises, and the
tie-break is deliberately conservative - a tie is no evidence, so do not spend the saving.

Matching: llama-quantize compiles each pattern as a C++ std::regex (ECMAScript).  We emit
`^blk\\.N\\.ffn_(gate|up)_exps\\.weight$` style anchors so a layer-2 rule cannot leak onto
layer-20.

  python3 tools/qwen38/emit_tensor_types.py ALLOC.json -o /tmp/tensor_types.txt
  cat /tmp/tensor_types.txt

Then:
  llama-quantize --imatrix IMATRIX.gguf --tensor-type-file /tmp/tensor_types.txt \
                 MODEL-BF16.gguf MODEL-XQ-Q2.gguf Q8_0
"""
import argparse
import collections
import json
import sys

# allocator tier name -> ggml_type_name (parse_ggml_type is case-insensitive, so this is cosmetic)
GGML = {
    "iq2_xxs": "IQ2_XXS", "q2_0": "Q2_0", "iq2_xs": "IQ2_XS", "iq2_s": "IQ2_S",
    "iq3_xxs": "IQ3_XXS", "iq3_s": "IQ3_S", "iq4_nl": "IQ4_NL",
    "ptq1": "PTQ1_0", "q3_k": "Q3_K", "q8_0": "Q8_0",
}
BPW = {
    "iq2_xxs": 2.0625, "q2_0": 2.25, "iq2_xs": 2.3125, "iq2_s": 2.5,
    "iq3_xxs": 3.0625, "iq3_s": 3.4375, "iq4_nl": 4.5,
    "ptq1": 1.75, "q3_k": 3.4375, "q8_0": 8.5,
}
PART_RE = {"gu": r"ffn_(gate|up)_exps", "dn": r"ffn_down_exps", "all": r"ffn_(gate|up|down)_exps"}


def collapse(per_expert) -> str:
    """One tier for a whole projection tensor, from that tensor's per-expert tiers."""
    if not per_expert:
        return None
    counts = collections.Counter(per_expert)
    if len(counts) == 1:
        return next(iter(counts))
    # most experts wins; ties broken toward more bits (do not spend an unearned saving)
    best = max(counts.items(), key=lambda kv: (kv[1], BPW.get(kv[0], 0.0)))
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alloc")
    ap.add_argument("-o", "--out", default="-")
    ap.add_argument("--per-proj", action="store_true",
                    help="the alloc map separates gate+up ('gu') from down ('dn')")
    ap.add_argument("--dense-type", default="Q8_0",
                    help="type for the non-expert tensors (default Q8_0; Strata requants dense at load)")
    ap.add_argument("--ptq1-placeholder", default="Q2_0",
                    help="type llama-quantize writes for a layer the allocation gives PTQ1_0. "
                         "It cannot emit 143 (outside the vendored ggml enum), so the tensor is "
                         "written at this type and apply_ternary.py replaces the payload afterwards.")
    ap.add_argument("--dry-run", action="store_true", help="print the per-layer table, not the file")
    a = ap.parse_args()
    if a.ptq1_placeholder.upper().startswith("PTQ1"):
        ap.error("--ptq1-placeholder cannot be PTQ1_0: llama-quantize has no such type")

    d = json.load(open(a.alloc))
    amap = d["map"]
    lines, table = [], []

    if a.per_proj:
        parts = [("gu", "gu"), ("dn", "dn")]
    else:
        parts = [("all", "all")]

    for layer in sorted(amap, key=int):
        exps = amap[layer]
        for part, key in parts:
            tiers = [e[key] for e in exps.values() if key in e]
            tier = collapse(tiers)
            if tier is None:
                continue
            pattern = r"^blk\.%s\.%s\.weight$" % (layer, PART_RE[part])
            emitted = a.ptq1_placeholder if tier == "ptq1" else GGML[tier]
            lines.append(f"{pattern}={emitted}")
            table.append((int(layer), part, tier, len(tiers), collections.Counter(tiers).most_common(3)))

    # The n-gram embedding table is 51.2B parameters / 26.8 GiB - 42% of the output file - and
    # it is a LOOKUP held off the accelerator (arXiv 2608.30320), so its precision costs RAM
    # but no compute.  It does not start with `blk.`, so the catch-all below would leave it on
    # the dense default; with --allow-requantize that would re-expand it to ~51 GiB and blow
    # the size gate.  Rules are first-match-wins, so this only has to precede the catch-all.
    lines.append("^per_layer_token_embd\\.weight$=IQ4_NL")

    # everything that is not a routed expert: dense, attention, GDN, shared expert, router, norms.
    # llama-quantize leaves non-quantisable tensors (norms, routers) at F32 on its own.
    # the lookahead must sit on the layer id, right after `blk.`: with `.*` before it, greedy
    # matching reaches the final dot ("weight") and the assertion can never see the expert name.
    lines.append(f"^blk\\.(?!.*ffn_(gate|up|down)_exps\\.weight).*$={a.dense_type}")

    if a.dry_run:
        print(f"{'L':>3} {'part':<5} {'tier':<9} {'n':>4}  distribution")
        for L, part, tier, n, dist in table:
            print(f"{L:>3} {part:<5} {tier:<9} {n:>4}  {dist}")
        tot = sum(len(amap[L]) * (2 if a.per_proj else 1) for L in amap)
        mixed = sum(1 for L in amap for e in amap[L].values()
                    for k in ("gu", "dn") if k in e and len(set(x[k] for x in amap[L].values() if k in x)) > 1)
        print(f"\n{len(amap)} layers, {tot} allocator decisions, "
              f"{mixed} layer/part groups needed a collapse (had >1 distinct tier)")
        return 0

    out = "\n".join(lines) + "\n"
    if a.out == "-":
        sys.stdout.write(out)
    else:
        open(a.out, "w").write(out)
        print(f"wrote {a.out}: {len(lines)} patterns "
              f"({len(lines)-2} expert layer/parts + PLE + 1 dense catch-all)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
