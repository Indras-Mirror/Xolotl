#!/usr/bin/env python3
"""Write a safetensors index that only references the shards we actually downloaded.

The converter derives its shard list FROM the index (`conversion/base.py:246` - part_names is
built from weight_map.values()) and then opens every one of them.  We deliberately skip the 31
pure-PLE shards, so an unmodified index makes it fail on files that are not there.

This drops exactly the tensors that live in a skipped shard.  Every tensor in a *kept* shard stays
in the map, so `tensor_names_from_index` still equals `tensor_names_from_parts` and the converter's
consistency check (`base.py:282`) passes.

Note the PLE tensors that live in the two MIXED shards (model-00005, model-00037) are kept - they
are on disk, and they carry the hash constants and row width the metadata needs.  The table itself
is incomplete (4 of 128 shards), which is why conversion also needs QWEN_PLE_DEFERRED=1; see
convert_ple_deferred.py.

  python3 tools/qwen38/filter_index.py --model-dir "/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8"
  python3 tools/qwen38/filter_index.py --model-dir ... --restore     # put the original back

The original is kept as model.safetensors.index.json.orig and is what --restore copies back.
Exit 0 = the filtered index was written and every referenced shard exists on disk.
"""
import argparse
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--index", default=str(HERE / "fp8-index.json"),
                    help="the ORIGINAL index (default: the copy kept with these tools)")
    ap.add_argument("--skipped", default=str(HERE / "skipped-ple-shards.json"))
    ap.add_argument("--restore", action="store_true", help="restore the original index and exit")
    a = ap.parse_args()

    md = pathlib.Path(a.model_dir)
    target = md / "model.safetensors.index.json"
    backup = md / "model.safetensors.index.json.orig"

    if a.restore:
        if not backup.is_file():
            print(f"no backup at {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, target)
        print(f"restored {target} from {backup}")
        return 0

    if not md.is_dir():
        print(f"model dir {md} does not exist", file=sys.stderr)
        return 1

    orig = json.load(open(a.index))
    weight_map = orig["weight_map"]
    skipped = set(json.load(open(a.skipped)))

    kept = {n: s for n, s in weight_map.items() if s not in skipped}
    dropped = len(weight_map) - len(kept)
    if dropped == 0:
        print("nothing to drop - is the skipped list empty?", file=sys.stderr)
        return 1

    # every referenced shard must be on disk, or the converter dies on open
    missing = sorted({s for s in kept.values() if not (md / s).is_file()})
    if missing:
        print(f"{len(missing)} referenced shard(s) are not on disk, e.g. {missing[:3]}\n"
              f"  finish the download first (bash tools/qwen38/fetch_fp8.sh \"{md}\")",
              file=sys.stderr)
        return 1

    if not backup.is_file():
        if target.is_file():
            shutil.copy2(target, backup)
            print(f"saved original index -> {backup.name}")
        else:
            print(f"warning: {target.name} was absent; nothing to back up", file=sys.stderr)

    out = {
        "metadata": {
            "total_size": orig.get("metadata", {}).get("total_size"),
            "note": ("PLE table deferred: tensors living in pure-PLE shards are omitted. "
                     "Convert with QWEN_PLE_DEFERRED=1 and serve with --ple-gguf."),
            "dropped_tensors": dropped,
            "dropped_shards": len(skipped),
        },
        "weight_map": kept,
    }
    target.write_text(json.dumps(out, indent=2))

    shards = len(set(kept.values()))
    print(f"wrote {target}")
    print(f"  tensors {len(kept)} (dropped {dropped} PLE)   shards {shards} "
          f"(dropped {len(skipped)} pure-PLE)")
    print(f"  all referenced shards present: yes")
    print(f"  restore with: python3 {pathlib.Path(__file__).name} "
          f"--model-dir \"{md}\" --restore")
    return 0


if __name__ == "__main__":
    sys.exit(main())
