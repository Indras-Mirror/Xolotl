#!/usr/bin/env python3
"""Prove a fetch manifest covers every tensor the model needs.

Compares the model's own safetensors index against the fetch manifest and the skip list, and
fails if any tensor would come from a shard that is neither fetched nor deliberately skipped.

This exists because a shard can contain BOTH the PLE table and ordinary layer tensors: the first
version of this pipeline classified `model-00037` as "PLE-only" (it has a `.ple.` tensor and no
experts) and dropped a whole layer's attention/GDN/hyper-connection weights.  A shard is only
skippable when EVERY tensor in it is PLE.

  python3 tools/qwen38/check_coverage.py [--index INDEX.json] [--manifest MANIFEST.json]
                                         [--skipped SKIPPED.json]
Exit 0 = every tensor is accounted for.
"""
import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def is_ple(name):
    return ".ple." in name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(HERE / "fp8-index.json"))
    ap.add_argument("--manifest", default=str(HERE / "fp8-manifest.json"))
    ap.add_argument("--skipped", default=str(HERE / "skipped-ple-shards.json"))
    a = ap.parse_args()

    w = json.load(open(a.index))["weight_map"]
    manifest = json.load(open(a.manifest))
    skipped = set(json.load(open(a.skipped)))

    by_shard = collections.defaultdict(list)
    for name, shard in w.items():
        by_shard[shard].append(name)

    problems = []

    # 1. every skipped shard must be pure PLE
    for shard in skipped:
        if shard not in by_shard:
            problems.append(f"skipped shard {shard} is not in the index at all")
            continue
        non_ple = [n for n in by_shard[shard] if not is_ple(n)]
        if non_ple:
            ex = ", ".join(sorted({n.split(".")[-2] for n in non_ple})[:6])
            problems.append(
                f"skipped shard {shard} is NOT pure PLE: {len(non_ple)} non-PLE tensors ({ex})")

    # 2. every non-skipped shard must be in the manifest
    for shard in by_shard:
        if shard in skipped:
            continue
        if shard not in manifest:
            problems.append(f"shard {shard} holds {len(by_shard[shard])} tensors but is not fetched")

    # 3. the counts must add up
    fetched = sum(len(by_shard[s]) for s in by_shard if s in manifest)
    dropped = sum(len(by_shard[s]) for s in skipped if s in by_shard)
    total = sum(len(v) for v in by_shard.values())
    if fetched + dropped != total:
        problems.append(f"tensor accounting: fetched {fetched} + skipped {dropped} != total {total}")

    experts = [n for n in w if "experts" in n]
    missing_experts = [n for n in experts if w[n] not in manifest]
    if missing_experts:
        problems.append(f"{len(missing_experts)} EXPERT tensors are not fetched, e.g. "
                        f"{missing_experts[:3]}")

    print(f"tensors total {total}  fetched {fetched}  skipped(PLE) {dropped}")
    print(f"expert tensors {len(experts)}  all fetched: {not missing_experts}")
    print(f"shards fetched {len(manifest) - 11} (approx)  skipped {len(skipped)}")
    if problems:
        print("\nFAIL:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nOK: every tensor is either fetched or in a pure-PLE shard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
