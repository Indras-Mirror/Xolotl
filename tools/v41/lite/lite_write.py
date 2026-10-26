#!/usr/bin/env python3
"""lite_write.py - write an HXQ3K-lite GGUF: the HXQ3K file (dense half by Xolotl's recipe from the heretic FP8
checkpoint, routed experts = llama.cpp imatrix Q3_K bytes) with the experts a lite_alloc.py map names replaced by
lite_calib.py's GPTQ Q2_0 bytes - gate+up and down independently (per-expert classes, ds41_geometry.hpp mixed path:
blk.L.ffn_<role>_exps.<type>.weight + I32 blk.L.ffn_exps_{gu,down}_type).

    V41_SRC=<heretic FP8 dir with .ok markers> python3 lite_write.py --map MAP.json --q2 OUT/q2 --out FILE.gguf
           [--layers A-B] [--dry-run]
    ... --verify FILE.gguf --map MAP.json --q2 OUT/q2 [--layers A-B]
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("V41_SRC_NAME", "DeepSeek V4.1 Flash 1017 heretic (pqhaz) - Xolotl HXQ3K-lite (Q3_K + GPTQ Q2_0 experts)")
os.environ.setdefault("V41_SRC_URL", "https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic")
import numpy as np  # noqa: E402
import v41_formats as F  # noqa: E402
import v41_gguf as W  # noqa: E402
import v41_gguf_meta as MD  # noqa: E402

from lite_config import Q3K_GGUF as HERETIC_GGUF, ENGRAM
TIER = {"q3_k": F.Q3_K, "q2_0": F.Q2_0}
ENGRAM_TABLES = {f"layers.{layer}.": ENGRAM.get(f"model-{shard:05d}-of-00048.safetensors")
                 for layer, shard in ((1, 47), (14, 48))}
_info = W.Source.info


def _info_engram(self, name):
    if ".engram.embed." in name:
        h, _ = self._shard(self.wm[name])
        if name not in h:
            import struct
            p = next(v for k, v in ENGRAM_TABLES.items() if name.startswith(k))
            if not p:
                raise ValueError("Missing Engram table header; set LITE_ENGRAM_L1 and LITE_ENGRAM_L14")
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                e = json.loads(f.read(n))[name]
            return e["dtype"], list(e["shape"]), -1, e["data_offsets"][1] - e["data_offsets"][0]
    return _info(self, name)


W.Source.info = _info_engram


class LiteMix:
    """(layer, expert, role) -> bytes: Q3_K from the heretic llama.cpp GGUF, or Q2_0 from lite_calib's q2/LNN.bin."""

    def __init__(self, mapfile, q2dir, dim=5120, inter=2304, ne=384, n_layers=40):
        self.map = json.load(open(mapfile))
        self.q2dir, self.ne, self.dim, self.inter = q2dir, ne, dim, inter
        kv, ts, start = W.read_gguf(HERETIC_GGUF)
        T = {n: (s, t, o) for n, s, t, o in ts}
        self.off, self.per = {}, {}
        for L in range(n_layers):
            for role in ("gate", "up", "down"):
                s, t, o = T[f"blk.{L}.ffn_{role}_exps.weight"]
                want = (inter, dim, ne) if role == "down" else (dim, inter, ne)
                if tuple(s) != want or t != F.Q3_K:
                    raise ValueError(f"blk.{L}.ffn_{role}_exps.weight: {s} {t}")
                self.off[(L, role)] = start + o
                self.per[(L, role)] = F.nbytes(t, s[:2])
        self.fd = os.open(HERETIC_GGUF, os.O_RDONLY)
        self.imatrix = os.path.basename(kv.get("quantize.imatrix.file", ""))
        self.q2fd = {}
        g = F.nbytes(F.Q2_0, (dim, inter))
        d = F.nbytes(F.Q2_0, (inter, dim))
        self.q2off = {"gate": (0, g), "up": (g, g), "down": (2 * g, d)}
        self.q2exb = 2 * g + d

    def ident(self):
        return "lite:" + self.map.get("name", "?") + ":" + self.map.get("sha", "")

    def types(self, L):
        m = self.map["layers"][str(L)]
        return [TIER[x] for x in m["gu"]], [TIER[x] for x in m["dn"]]

    def get(self, L, e, role, src=None):
        m = self.map["layers"][str(L)]
        tier = m["dn" if role == "down" else "gu"][e]
        if tier == "q3_k":
            per = self.per[(L, role)]
            b = os.pread(self.fd, per, self.off[(L, role)] + e * per)
            if len(b) != per:
                raise OSError(f"short read L{L} e{e} {role}")
            return b
        fd = self.q2fd.get(L)
        if fd is None:
            p = os.path.join(self.q2dir, f"L{L:02d}.bin")
            if os.path.getsize(p) != self.ne * self.q2exb:
                raise ValueError(f"{p}: size {os.path.getsize(p)}")
            fd = self.q2fd[L] = os.open(p, os.O_RDONLY)
        o, n = self.q2off[role]
        b = os.pread(fd, n, e * self.q2exb + o)
        if len(b) != n:
            raise OSError(f"short q2 read L{L} e{e} {role}")
        return b


def records_for(provider, plan, layers, partial):
    _, records = MD.metadata(W.MAIN, W.NAME, W.URL)
    counts = {}
    for t in plan:
        if t.kind == "experts" and t.role != "up":
            k = ("gu" if t.role == "gate" else "down", F.TYPE_NAME[t.qtype])
            counts[k] = counts.get(k, 0) + len(t.experts)
    m = provider.map
    records += [
        MD.kv_string("deepseek41.quantization", "routed experts: Q3_K (llama.cpp imatrix) + GPTQ Q2_0 (Xolotl v41 grids, "
                     "own calibration) chosen per expert and projection by held-out output error; Q8_0 attention / "
                     "shared / head; F16 embedding"),
        MD.kv_string("strata.v41.expert_source", provider.ident()),
        MD.kv_string("strata.v41.expert_tiers", json.dumps({f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())})),
        MD.kv_string("strata.v41.expert_layout", "uniform role: ffn_<role>_exps.weight; mixed: ffn_<role>_exps.<type>.weight "
                     "per type (experts in id order) + I32 ffn_exps_{gu,down}_type[384]"),
        MD.kv_string("strata.v41.expert_imatrix", provider.imatrix),
        MD.kv_string("strata.v41.lite_alloc", json.dumps({k: v for k, v in m.items() if k != "layers"})),
    ]
    if partial:
        records.append(MD.kv_u32_array("strata.v41.partial_layers", layers))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--q2", required=True)
    ap.add_argument("--out")
    ap.add_argument("--verify")
    ap.add_argument("--layers")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = W.Source()
    cfg = json.load(open(os.path.join(W.MAIN, "config.json")))
    c = cfg["text_config"]
    layers = W.parse_layers(a.layers, c["num_hidden_layers"])
    pv = LiteMix(a.map, a.q2, dim=c["hidden_size"], inter=c["moe_intermediate_size"], ne=c["n_routed_experts"],
                 n_layers=c["num_hidden_layers"])
    if a.verify:
        n, bad = W.verify(a.verify, src, pv, cfg, layers)
        print(f"v41_gguf.verify: {n} tensors checked, {len(bad)} problems")
        for b in bad[:20]:
            print("  " + b)
        sys.exit(1 if bad else 0)
    if not a.out:
        ap.error("--out or --verify")
    plan, disk, claimed = W.build_plan(cfg, src, pv.types, layers, engram=False)
    assert not disk
    if a.layers is None:
        omitted = {n for n in src.wm if n.startswith(("mtp.", "vision.", "aligner.", "image_"))}
        rest = set(src.wm) - claimed - omitted
        if rest:
            raise ValueError(f"unclaimed source tensors: {sorted(rest)[:10]}")
    records = records_for(pv, plan, layers, a.layers is not None)
    hdr = W.header(plan, records)
    size = len(hdr) + sum(W.align(t.nbytes) for t in plan)
    ex = sum(t.nbytes for t in plan if t.kind == "experts")
    W.log(f"plan: {len(plan)} tensors, file {size / 2**30:.2f} GiB ({size / 1e9:.1f} GB): experts {ex / 2**30:.2f} GiB, "
          f"dense {(size - ex) / 2**30:.2f} GiB; layers {layers[0]}-{layers[-1]}")
    if a.dry_run:
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    W.write_shard(a.out, plan, records, src, pv, 1, pv.ident())
    W.experts_json(a.out, len(hdr), plan)
    W.log(f"wrote {a.out} ({size / 2**30:.2f} GiB) in {time.time() - t0:.0f} s + {a.out}.experts.json")


if __name__ == "__main__":
    main()
