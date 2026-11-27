"""dfly2_sidecar.py - the DFly2 block drafter for DeepSeek-V4.1-Flash (AngelSpec dfly2-b6) as a sidecar GGUF (CPU only).

    python3 tools/v41/dfly2_sidecar.py --src DIR/dfly2-b6 --out DIR/DFly2-B6-ds41.gguf
    python3 tools/v41/dfly2_sidecar.py --src DIR/dfly2-b6 --verify DIR/DFly2-B6-ds41.gguf [--sample 4]

Same packing rules as v41_sidecar.py (the DSpark sidecar), HF names kept (minus the leading "model."):
  - routed experts (FP4 e2m1 + E8M0 per 32) -> MXFP4, a lossless byte repack, stacked per role:
    layers.N.ffn.experts.{w1,w2,w3}.weight  ggml [in, out, n_expert]
  - every other 2-D BF16 weight with a row length divisible by 32 -> Q8_0
  - norms, biases, router bias, mHC, sinks, gates, block biases -> F32
The draft config (block size, mask id, target layers, rope) goes in as metadata (dfly2.*).
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "third_party", "llama.cpp", "gguf-py"))
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

from v41_gguf import Source  # noqa: E402
from v41_sidecar import MX, Q8, ROLES, fp4_to_mxfp4, log  # noqa: E402


class DirSource(Source):
    """A plain safetensors dir (no .ok markers)."""

    def _shard(self, shard):
        if shard not in self.hdr:
            path = os.path.join(self.dirs[0], shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            self.hdr[shard] = (h, 8 + n)
            self.fds[shard] = os.open(path, os.O_RDONLY)
        return self.hdr[shard]


def gname(hf):
    return hf.removeprefix("model.")


def plan(src):
    """[(gguf name, kind, payload)] in write order.  kind: 'mx' (stacked experts), 'q8', 'f32'."""
    wm = src.wm
    out = []
    for n in sorted(wm):
        if ".experts." in n or n.endswith(".scale") or n.endswith(".bias_vl"):
            continue
        dt, shape, _, _ = src.info(n)
        kind = "q8" if len(shape) == 2 and shape[1] % 32 == 0 and shape[0] >= 32 and dt != "F32" else "f32"
        out.append((gname(n), kind, n))
    layers = sorted({int(n.split(".")[2]) for n in wm if n.startswith("model.layers.")})
    for l in layers:
        p = f"model.layers.{l}.ffn.experts."
        ne = 1 + max(int(n.split(".")[5]) for n in wm if n.startswith(p))
        for r in ROLES:
            out.append((f"layers.{l}.ffn.experts.{r}.weight", "mx", [f"{p}{e}.{r}.weight" for e in range(ne)]))
    return out


def tensor_info(src, kind, pay):
    if kind == "mx":
        dt, (R, Ch), _, _ = src.info(pay[0])
        shape = (len(pay), R, Ch * 2 // 32 * 17)
        return shape, np.uint8, int(np.prod(shape)), MX
    dt, shape, _, _ = src.info(pay)
    if kind == "q8":
        R, C = shape
        return (R, C // 32 * 34), np.uint8, R * C // 32 * 34, Q8
    return tuple(shape), np.float32, int(np.prod(shape)) * 4, None


def tensor_data(src, kind, pay):
    if kind == "mx":
        return np.stack([fp4_to_mxfp4(src, n) for n in pay])
    v = src.f32(pay)
    if kind == "q8":
        return quants.quantize(v.astype(np.float32), Q8)
    return np.ascontiguousarray(v, np.float32)


def write(args):
    src = DirSource([args.src])
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    dc = cfg["dfly_config"]
    base = {}
    if args.draft_config and os.path.exists(args.draft_config):
        base = json.load(open(args.draft_config))
    pl = plan(src)
    w = gguf.GGUFWriter(args.out + ".tmp", "deepseek41-dfly2")
    w.add_name("DFly2 block drafter for DeepSeek-V4.1-Flash (AngelSpec dfly2-b6, Xolotl sidecar)")
    w.add_uint32("dfly2.block_size", int(dc["block_size"]))
    w.add_uint32("dfly2.mask_token_id", int(dc["mask_token_id"]))
    w.add_array("dfly2.target_kv_layers", [int(x) for x in dc["target_kv_layers"]])
    w.add_array("dfly2.target_index_layers", [int(x) for x in dc["target_index_layers"]])
    w.add_uint32("dfly2.index_topk", int(dc["index_topk"]))
    w.add_uint32("dfly2.local_window", int(base.get("context_local_window", 128)))
    w.add_uint32("dfly2.n_expert", int(base.get("num_experts", 128)))
    w.add_uint32("dfly2.n_expert_used", int(base.get("num_experts_per_tok", 3)))
    w.add_float32("dfly2.routed_scale", float(base.get("router_scaling_factor", 1.5)))
    w.add_float32("dfly2.swiglu_limit", float(base.get("swiglu_limit", 10.0)))
    w.add_uint32("dfly2.hc_mult", int(base.get("hc_mult", 4)))
    w.add_uint32("dfly2.hc_sinkhorn_iters", int(base.get("hc_sinkhorn_iters", 20)))
    w.add_float32("dfly2.hc_eps", float(base.get("hc_eps", 1e-6)))
    r = dc["rope"]
    w.add_float32("dfly2.rope.theta", float(r["theta"]))
    w.add_float32("dfly2.rope.factor", float(r["factor"]))
    w.add_uint32("dfly2.rope.original_seq_len", int(r["original_seq_len"]))
    w.add_float32("dfly2.rope.beta_fast", float(r["beta_fast"]))
    w.add_float32("dfly2.rope.beta_slow", float(r["beta_slow"]))
    w.add_string("dfly2.config_json", json.dumps(dc))
    w.add_string("dfly2.experts_type", "MXFP4 (lossless repack of the checkpoint FP4)")
    total = 0
    for name, kind, pay in pl:
        shape, dt, nb, raw = tensor_info(src, kind, pay)
        w.add_tensor_info(name, shape, np.dtype(dt), nb, raw_dtype=raw)
        total += nb
    log(f"{len(pl)} tensors, {total / 1e9:.2f} GB -> {args.out}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for i, (name, kind, pay) in enumerate(pl):
        a = tensor_data(src, kind, pay)
        w.write_tensor_data(a)
        if kind == "mx" or i % 20 == 0:
            log(f"  [{i + 1}/{len(pl)}] {name} {kind} {a.nbytes / 1e6:.1f} MB")
    w.close()
    os.replace(args.out + ".tmp", args.out)
    log("done")


def verify(args):
    """Experts: dequantized MXFP4 == the checkpoint's FP4 values exactly.  Q8_0: relative RMS error.  F32: exact."""
    src = DirSource([args.src])
    r = gguf.GGUFReader(args.verify)
    rng = np.random.default_rng(0)
    worst, n_mx, n_q8, n_f = 0.0, 0, 0, 0
    for t in r.tensors:
        if t.tensor_type == MX:
            ne = int(t.shape[2])
            for e in rng.choice(ne, min(args.sample, ne), replace=False):
                hf = "model." + t.name.replace("experts.", f"experts.{e}.")
                ref = src.f32(hf)
                got = quants.dequantize(np.asarray(t.data[e]), MX).reshape(ref.shape)
                if not np.array_equal(got, ref):
                    raise SystemExit(f"MISMATCH {hf}: max abs {np.abs(got - ref).max()}")
                n_mx += 1
        elif t.tensor_type == Q8:
            ref = src.f32("model." + t.name)
            got = quants.dequantize(np.asarray(t.data), Q8).reshape(ref.shape)
            rel = float(np.sqrt(np.mean((got - ref) ** 2)) / max(np.sqrt(np.mean(ref ** 2)), 1e-30))
            worst = max(worst, rel)
            n_q8 += 1
        else:
            ref = src.f32("model." + t.name)
            got = np.asarray(t.data, np.float32).reshape(ref.shape)
            if not np.array_equal(got, ref):
                raise SystemExit(f"MISMATCH {t.name} (F32)")
            n_f += 1
    print(f"verify: {n_mx} MXFP4 experts bit-exact vs FP4; {n_f} F32 exact; {n_q8} Q8_0 tensors, "
          f"worst relative RMS error {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--draft-config", default=os.path.expanduser(
        "~/.quetza-data/dfly2/src/AngelSpec/angelspec/config/dfly2_dsv41_flash_draft_config.json"))
    ap.add_argument("--out")
    ap.add_argument("--verify")
    ap.add_argument("--sample", type=int, default=4)
    args = ap.parse_args()
    if args.verify:
        verify(args)
    elif args.out:
        write(args)
    else:
        ap.error("--out or --verify")


if __name__ == "__main__":
    main()
