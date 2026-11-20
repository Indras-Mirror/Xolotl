#!/usr/bin/env python3
"""Check a LoRA GGUF against a base model the way llama.cpp's loader will (src/llama-adapter.cpp:195-390).

usage: lora_loader_check.py <model.gguf> <adapter.gguf> [...]

Replicates the checks that decide whether llama.cpp accepts or throws:
  * general.type == "adapter", adapter.type == "lora"
  * general.architecture must equal the model's (llama_arch_from_string -> model.arch)
  * every tensor name is a real base-model tensor (else: "does not exist in base model")
  * each .lora_a has a .lora_b ("pair is missing one component")
  * non-embedding: model.ne[0] == a.ne[0] and model.ne[1] == b.ne[1]
  *              and a.ne[1] == b.ne[0]  ("lora_a tensor is not transposed")
The applied delta is scale * B(A x) with scale = alpha/rank, rank = b.ne[0] (llama-adapter.h:53).
"""
import sys

sys.path.insert(0, "/media/Crucial1TB/xolotl-train/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402


def kv(reader, key):
    f = reader.fields.get(key)
    return f.contents() if f else None


def main(model_path, adapter_paths):
    model = GGUFReader(model_path)
    tensors = {t.name: t for t in model.tensors}
    arch = kv(model, "general.architecture")
    print(f"base: {model_path}  arch={arch}  tensors={len(tensors)}")

    rc = 0
    for ap in adapter_paths:
        ad = GGUFReader(ap)
        problems = []
        gtype, atype = kv(ad, "general.type"), kv(ad, "adapter.type")
        aarch, alpha = kv(ad, "general.architecture"), kv(ad, "adapter.lora.alpha")
        if gtype != "adapter":
            problems.append(f"general.type is {gtype!r}, loader wants 'adapter'")
        if atype != "lora":
            problems.append(f"adapter.type is {atype!r}, loader wants 'lora'")
        if aarch != arch:
            problems.append(f"arch mismatch: adapter {aarch!r} vs model {arch!r} -> 'model arch and LoRA arch mismatch'")

        pairs = {}
        for t in ad.tensors:
            if t.name.endswith(".lora_a"):
                pairs.setdefault(t.name[:-7], {})["a"] = t
            elif t.name.endswith(".lora_b"):
                pairs.setdefault(t.name[:-7], {})["b"] = t
            else:
                problems.append(f"tensor {t.name!r} has an unexpected suffix")

        print(f"\nadapter: {ap}  ({len(pairs)} weights, alpha={alpha})")
        for name in sorted(pairs):
            pa = pairs[name]
            a, b = pa.get("a"), pa.get("b")
            if a is None or b is None:
                problems.append(f"{name}: pair missing one component")
                continue
            mt = tensors.get(name)
            if mt is None:
                problems.append(f"{name}: does not exist in base model")
                continue
            ok_shape = mt.shape[0] == a.shape[0] and mt.shape[1] == b.shape[1]
            ok_rank = a.shape[1] == b.shape[0]
            scale = (alpha / b.shape[0]) if alpha else 1.0
            if not ok_shape:
                problems.append(f"{name}: shape W{list(mt.shape)} a{list(a.shape)} b{list(b.shape)}")
            if not ok_rank:
                problems.append(f"{name}: lora_a not transposed (a.ne1 {a.shape[1]} != b.ne0 {b.shape[0]})")
            print(f"  {name:26s} W{str(list(mt.shape)):16s} a{str(list(a.shape)):11s} b{str(list(b.shape)):11s} "
                  f"scale {scale:.3f}  {'OK' if ok_shape and ok_rank else 'REJECT'}")
        if problems:
            rc = 1
            print(f"  -> llama.cpp would REJECT: {problems[0]}" + (f" (+{len(problems)-1} more)" if len(problems) > 1 else ""))
        else:
            print("  -> llama.cpp loader checks: PASS")
    return rc


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2:]))
