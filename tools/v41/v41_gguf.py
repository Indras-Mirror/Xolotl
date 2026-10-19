"""v41_gguf.py - write the DeepSeek-V4.1-Flash mixed-tier GGUF (`deepseek41`) for Strata-DS4, straight from the HF
FP8/FP4 shards (CPU only, streaming, resumable).

    python3 tools/v41/v41_gguf.py --out DIR/V41-GSQ.gguf --experts QDIR            # the real export
    python3 tools/v41/v41_gguf.py --out DIR/V41-Q2.gguf --rtn q2_0                  # RTN control / fallback
    python3 tools/v41/v41_gguf.py --out ... --rtn-map alloc.json                    # RTN at a v41_alloc tier map
    python3 tools/v41/v41_gguf.py --out ... --rtn-map alloc.json --experts-in-place --no-engram
                         # the final file without a QDIR copy: dense + type maps now, expert regions left as holes
                         # (+ <shard>.experts.json offsets) that v41_quant.py layer --gguf fills layer by layer
    python3 tools/v41/v41_gguf.py --verify DIR/V41-GSQ-00001-of-00002.gguf [--experts QDIR | --rtn ...]
    options: --layers A-B (partial file: globals + those blocks; tests), --no-engram, --engram-rows N (tests),
             --engram-dir D (shard 2 written there, symlinked next to shard 1), --threads N, --dry-run

Two shards, llama.cpp split naming (strata::gguf_split_paths reads them as one model):
  <stem>-00001-of-00002.gguf  everything except the Engram tables
  <stem>-00002-of-00002.gguf  `blk.{1,14}.engram_embd.weight`: the FP8 rows + their E8M0 scales byte for byte
                              (264 B per row, DwarfStar's e4m3_e8m0_32_row264), ~189 GiB.  Read by row, never resident;
                              its own shard so it can sit on another SSD (symlink it).
Dense half: DwarfStar's tensor names, shapes and types (deepseek41_quantize.py @fc80bd6): Q8_0 attention / shared
expert / head, F16 embedding + compressor/indexer + Engram wkv, F32 norms / router / mHC.  Metadata: v41_gguf_meta.

Routed experts, per layer.  Gate and up share one type per expert (the grouped kernel's gu_type), down has its own
(d_type).  When every expert of a layer has the same type for a role, the role is the usual stacked tensor
(`blk.L.ffn_gate_exps.weight`, [n_embd, n_ff, 384]).  Otherwise each type present gets its own stacked tensor of the
experts at that type, in expert order: `blk.L.ffn_{gate,up}_exps.<type>.weight` [n_embd, n_ff, n_t] and
`blk.L.ffn_down_exps.<type>.weight` [n_ff, n_embd, n_t], and I32 maps `blk.L.ffn_exps_gu_type` /
`blk.L.ffn_exps_down_type` [384] give each expert's ggml type (its slot = its rank among the experts of that type).
<type> is q2_0 / q3_k / ptq1_0 (ids 42 / 11 / 143; 143 is llama-tq's PTQ1_0, not upstream ggml).

Expert sources:
  --experts QDIR   the quantize pass's output: QDIR/LNN.json {"gu_type": [384 ids], "d_type": [384 ids]} and
                   QDIR/LNN.bin = for each expert in order its gate rows, up rows, down rows (GGUF block bytes,
                   v41_formats.encode_tier).  The writer only copies.
  --rtn TIER       round-to-nearest from the FP4 source (v41_formats.encode_rtn), every expert at TIER.
  --rtn-map J      RTN at the tiers of a v41_alloc.py --out map (layers missing from it: --rtn's tier, default q2_0).
"""
import argparse
import concurrent.futures as cf
import dataclasses
import hashlib
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v41_formats as F  # noqa: E402
import v41_gguf_meta as MD  # noqa: E402

# V41_SRC="dir1:dir2:..." = another checkpoint (see v41_calib.py); V41_SRC_NAME / V41_SRC_URL = its GGUF metadata name
# and HF page.  Unset = the dealignai FP8.
_SRC = [d for d in os.environ.get("V41_SRC", "").split(os.pathsep) if d]
MAIN = _SRC[0] if _SRC else "/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"
DIRS = _SRC or [MAIN, "/media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8",
                "/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"]          # ~ v41_calib.DIRS
NAME = os.environ.get("V41_SRC_NAME", "DeepSeek V4.1 Flash UNCENSORED (Strata v41 GSQ-RCO mixed tier)")
URL = os.environ.get("V41_SRC_URL", "https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8")
ROLES = (("gate", "w1"), ("up", "w3"), ("down", "w2"))
ENGRAM_ROW = 264
ALIGN = MD.GGUF_ALIGNMENT


def align(n, a=ALIGN):
    return (n + a - 1) // a * a


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ================================================================== source shards
if hasattr(os, "pread"):
    _pread = os.pread
else:   # Windows has no os.pread: seek + read under a lock (callers may read from threads)
    import threading
    _pread_lock = threading.Lock()

    def _pread(fd, n, off):
        with _pread_lock:
            os.lseek(fd, off, os.SEEK_SET)
            return os.read(fd, n)


class Source:
    """The HF checkpoint: header lookups and pread()s by tensor name, FP8 / FP4 dequantization."""

    def __init__(self, dirs=DIRS):
        self.dirs = dirs
        self.wm = json.load(open(os.path.join(dirs[0], "model.safetensors.index.json")))["weight_map"]
        self.hdr, self.fds = {}, {}
        self._fp8 = None

    def _shard(self, shard):
        if shard not in self.hdr:
            path = next((os.path.join(d, shard) for d in self.dirs if os.path.exists(os.path.join(d, shard) + ".ok")), None)
            if path is None:
                raise FileNotFoundError(f"shard {shard} is not verified (.ok) in any source dir")
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            self.hdr[shard] = (h, 8 + n)
            self.fds[shard] = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        return self.hdr[shard]

    def info(self, name):
        h, base = self._shard(self.wm[name])
        e = h[name]
        return e["dtype"], list(e["shape"]), base + e["data_offsets"][0], e["data_offsets"][1] - e["data_offsets"][0]

    def __contains__(self, name):
        return name in self.wm

    def read(self, name, start=0, n=None):
        _, _, off, size = self.info(name)
        n = size - start if n is None else n
        fd = self.fds[self.wm[name]]
        out, got = bytearray(n), 0
        while got < n:
            b = _pread(fd, min(n - got, 1 << 28), off + start + got)
            if not b:
                raise OSError(f"{name}: short read")
            out[got:got + len(b)] = b
            got += len(b)
        return bytes(out)

    def fp8_lut(self):
        if self._fp8 is None:
            try:
                import torch
                v = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
            except ImportError:   # the same 256 values without torch (checked equal to torch's, NaNs included)
                b = np.arange(256, dtype=np.int32)
                s, e, m = (b >> 7) & 1, (b >> 3) & 0xF, b & 7
                v = np.where(e == 0, (m / 8.0) * 2.0 ** -6, (1.0 + m / 8.0) * np.exp2(e.astype(np.float64) - 7.0))
                v = np.where(s == 1, -v, v).astype(np.float32)
                v[(e == 15) & (m == 7)] = np.nan
            self._fp8 = v
        return self._fp8

    @staticmethod
    def e8m0(b):
        s = np.frombuffer(b, np.uint8) if isinstance(b, (bytes, bytearray)) else b
        if np.any(s == 255):
            raise ValueError("nonfinite E8M0 scale")
        return np.ldexp(np.float32(1), s.astype(np.int32) - 127).astype(np.float32)

    def f32(self, name, r0=0, r1=None):
        """Rows [r0, r1) of tensor `name` as float32 (2-D: [rows, cols]; 1-D: the whole vector)."""
        dt, shape, _, _ = self.info(name)
        if len(shape) == 1:
            return self._plain(dt, self.read(name)).reshape(shape)
        R, C = shape
        r1 = R if r1 is None else min(r1, R)
        if dt == "F8_E4M3":
            assert r0 % 32 == 0
            codes = np.frombuffer(self.read(name, r0 * C, (r1 - r0) * C), np.uint8).reshape(r1 - r0, C)
            if np.any((codes & 127) == 127):
                raise ValueError(f"{name}: nonfinite FP8 weight")
            sn = name.removesuffix(".weight") + ".scale"
            sdt, sshape, _, _ = self.info(sn)
            assert sdt == "F8_E8M0" and sshape == [(R + 31) // 32, (C + 31) // 32], (sn, sdt, sshape)
            sr0, sr1 = r0 // 32, (r1 + 31) // 32
            sc = self.e8m0(self.read(sn, sr0 * sshape[1], (sr1 - sr0) * sshape[1])).reshape(sr1 - sr0, sshape[1])
            sce = np.repeat(np.repeat(sc, 32, 0), 32, 1)[: r1 - r0, :C]
            return self.fp8_lut()[codes] * sce
        if dt == "I8":   # FP4 e2m1, two per byte (low nibble first), E8M0 per 32
            Cw = C * 2
            codes = np.frombuffer(self.read(name, r0 * C, (r1 - r0) * C), np.uint8).reshape(r1 - r0, C)
            lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
            v = np.empty((r1 - r0, Cw), np.float32)
            v[:, 0::2], v[:, 1::2] = lut[codes & 15], lut[codes >> 4]
            sn = name.removesuffix(".weight") + ".scale"
            sdt, sshape, _, _ = self.info(sn)
            assert sdt == "F8_E8M0" and sshape == [R, Cw // 32], (sn, sdt, sshape)
            sc = self.e8m0(self.read(sn, r0 * sshape[1], (r1 - r0) * sshape[1])).reshape(r1 - r0, -1)
            return (v.reshape(r1 - r0, -1, 32) * sc[..., None]).reshape(r1 - r0, Cw)
        isz = {"BF16": 2, "F16": 2, "F32": 4}[dt]
        return self._plain(dt, self.read(name, r0 * C * isz, (r1 - r0) * C * isz)).reshape(r1 - r0, C)

    @staticmethod
    def _plain(dt, b):
        if dt == "BF16":
            return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32)
        if dt == "F16":
            return np.frombuffer(b, np.float16).astype(np.float32)
        if dt == "F32":
            return np.frombuffer(b, np.float32).copy()
        raise ValueError(f"unsupported source dtype {dt}")


# ================================================================== the plan
@dataclasses.dataclass
class T:
    name: str
    shape: tuple          # ggml order (ne0 first)
    qtype: int
    kind: str             # regular | experts | etype | engram
    src: str = ""         # regular/engram: source tensor; experts: source pattern with {e}
    layer: int = -1
    role: str = ""        # gate | up | down
    experts: tuple = ()   # experts in this stacked tensor (experts kind) / the per-expert types (etype kind)
    shard: int = 0
    offset: int = 0
    nbytes: int = 0


def build_plan(cfg, src, types_of, layers, engram=True, engram_rows=None):
    """types_of(L) -> (gu_types[384], d_types[384]).  Returns (main tensors, engram tensors, claimed source names)."""
    c = cfg["text_config"]
    if cfg["quantization_config"]["weight_block_size"] != [32, 32]:
        raise ValueError("expected native 32x32 FP8 blocks")
    dim, inter = c["hidden_size"], c["moe_intermediate_size"]
    heads, hd = c["num_attention_heads"], c["head_dim"]
    qrank, orank, groups = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    hc, NE, vocab = c["hc_mult"], c["n_routed_experts"], c["vocab_size"]
    ih, idim = c["index_n_heads"], c["index_head_dim"]
    plan, disk, claimed = [], [], set()

    def claim(name, shape, dtype=None):
        dt, sh, _, _ = src.info(name)
        if sh != list(shape) or (dtype and dt != dtype):
            raise ValueError(f"{name}: {dt} {sh}, expected {dtype} {list(shape)}")
        claimed.add(name)
        if dt in ("I8", "F8_E4M3"):
            claimed.add(name.removesuffix(".weight") + ".scale")

    def regular(name, source, shape, qtype):
        claim(source, shape)
        plan.append(T(name, tuple(reversed(shape)), qtype, "regular", source))

    regular("token_embd.weight", "embed.weight", (vocab, dim), F.F16)
    regular("output_norm.weight", "norm.weight", (dim,), F.F32)
    regular("output.weight", "head.weight", (vocab, dim), F.Q8_0)
    for L in layers:
        s, d = f"layers.{L}", f"blk.{L}"
        for site in ("attn", "ffn"):
            for part, shape, qt in (("fn", (hc * (hc + 2), hc * dim), F.F16), ("base", (hc * (hc + 2),), F.F32),
                                    ("scale", (3,), F.F32)):
                regular(f"{d}.hc_{site}_{part}.weight", f"{s}.hc_{site}_{part}", shape, qt)
            regular(f"{d}.{site}_norm.weight", f"{s}.{site}_norm.weight", (dim,), F.F32)
        for tgt, so, shape, qt in (
            ("attn_sinks.weight", "attn_sink", (heads,), F.F32),
            ("attn_q_a.weight", "wq_a.weight", (qrank, dim), F.Q8_0),
            ("attn_q_b.weight", "wq_b.weight", (heads * hd, qrank), F.Q8_0),
            ("attn_q_a_norm.weight", "q_norm.weight", (qrank,), F.F32),
            ("attn_kv.weight", "wkv.weight", (hd, dim), F.Q8_0),
            ("attn_kv_a_norm.weight", "kv_norm.weight", (hd,), F.F32),
            ("attn_output_a.weight", "wo_a.weight", (groups * orank, heads * hd // groups), F.Q8_0),
            ("attn_output_b.weight", "wo_b.weight", (dim, groups * orank), F.Q8_0),
        ):
            regular(f"{d}.{tgt}", f"{s}.attn.{so}", shape, qt)
        if L in c["kv_source_layer_ids"]:
            for tgt, so, shape, qt in (
                ("attn_compressor_kv.weight", "compressor.wkv.weight", (hd, dim), F.F16),
                ("attn_compressor_norm.weight", "compressor.norm.weight", (hd,), F.F32),
                ("indexer.attn_k.weight", "indexer.wk.weight", (idim, hd), F.F16),
                ("indexer.k_norm.weight", "indexer.k_norm.weight", (idim,), F.F32),
            ):
                regular(f"{d}.{tgt}", f"{s}.attn.{so}", shape, qt)
            if c["compress_ratios"][L] > 1:
                regular(f"{d}.attn_compressor_gate.weight", f"{s}.attn.compressor.wgate.weight", (hd, dim), F.F16)
        if L in c["index_source_layer_ids"]:
            regular(f"{d}.indexer.attn_q_b.weight", f"{s}.attn.indexer.wq_b.weight", (ih * idim, qrank), F.F16)
            regular(f"{d}.indexer.proj.weight", f"{s}.attn.indexer.weights_proj.weight", (ih, dim), F.F16)
        regular(f"{d}.ffn_gate_inp.weight", f"{s}.ffn.gate.weight", (NE, dim), F.F32)
        for suffix, tgt in (("bias", "exp_probs_b.bias"), ("bias_vl", "exp_probs_b_vl.bias")):
            regular(f"{d}.{tgt}", f"{s}.ffn.gate.{suffix}", (NE,), F.F32)
        for role, w in ROLES:
            shape = (inter, dim) if role != "down" else (dim, inter)
            regular(f"{d}.ffn_{role}_shexp.weight", f"{s}.ffn.shared_experts.{w}.weight", shape, F.Q8_0)
            for e in range(NE):
                claim(f"{s}.ffn.experts.{e}.{w}.weight", (shape[0], shape[1] // 2), "I8")
        gu, dn = types_of(L)
        assert len(gu) == NE and len(dn) == NE, L
        for role, w in ROLES:
            ty = dn if role == "down" else gu
            shape = (dim, inter) if role == "down" else (inter, dim)     # [rows, cols]
            uniq = sorted(set(ty))
            for t in uniq:
                if t not in F.TYPE_TIER:
                    raise ValueError(f"layer {L} {role}: type {t} is not an expert tier")
                ex = tuple(e for e in range(NE) if ty[e] == t)
                suffix = "" if len(uniq) == 1 else "." + F.TYPE_NAME[t].lower()
                plan.append(T(f"{d}.ffn_{role}_exps{suffix}.weight", (shape[1], shape[0], len(ex)), t, "experts",
                              f"{s}.ffn.experts.{{e}}.{w}.weight", L, role, ex))
        for key, ty in (("gu", gu), ("down", dn)):
            if len(set(ty)) > 1:
                plan.append(T(f"{d}.ffn_exps_{key}_type", (NE,), F.I32, "etype", layer=L, experts=tuple(ty)))
        if L in c["engram_layer_ids"]:
            rows = c["engram_num_embeddings"][c["engram_layer_ids"].index(L)]
            eg = f"{s}.engram"
            claim(f"{eg}.embed.weight", (rows, 256), "F8_E4M3")
            claim(f"{eg}.embed.scale", (rows, 8), "F8_E8M0")
            if engram:
                n = rows if engram_rows is None else min(rows, engram_rows)
                disk.append(T(f"{d}.engram_embd.weight", (ENGRAM_ROW, n), F.I8, "engram", f"{eg}.embed.weight",
                              L, shard=1))
            for part in ("q", "k"):
                regular(f"{d}.engram_{part}_norm.weight", f"{eg}.{part}_weight", (hc, dim), F.F32)
            regular(f"{d}.engram_kv.weight", f"{eg}.wkv.weight", ((hc + 1) * dim, 24 * 256), F.F16)
    for group in (plan, disk):
        off = 0
        for t in group:
            t.offset = off
            t.nbytes = F.nbytes(t.qtype, t.shape) if t.kind != "engram" else ENGRAM_ROW * t.shape[1]
            off += align(t.nbytes)
    return plan, disk, claimed


# ================================================================== expert providers
class ExpertDir:
    """The quantize pass's per-layer files (module docstring)."""

    def __init__(self, path, dim, inter):
        self.path, self.dim, self.inter = path, dim, inter
        self._idx = {}

    def ident(self):
        h = hashlib.sha256()
        for f in sorted(os.listdir(self.path)):
            p = os.path.join(self.path, f)
            st = os.stat(p)
            h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
        return "dir:" + h.hexdigest()

    def types(self, L):
        j = json.load(open(os.path.join(self.path, f"L{L:02d}.json")))
        return list(j["gu_type"]), list(j["d_type"])

    def _layer(self, L):
        if L not in self._idx:
            gu, dn = self.types(L)
            offs, o = [], 0
            for e in range(len(gu)):
                g = F.nbytes(gu[e], (self.dim, self.inter))
                dd = F.nbytes(dn[e], (self.inter, self.dim))
                offs.append((o, o + g, o + 2 * g, g, dd))
                o += 2 * g + dd
            p = os.path.join(self.path, f"L{L:02d}.bin")
            if os.path.getsize(p) != o:
                raise ValueError(f"{p}: {os.path.getsize(p)} bytes, the types say {o}")
            self._idx[L] = (offs, os.open(p, os.O_RDONLY | getattr(os, "O_BINARY", 0)))
        return self._idx[L]

    def get(self, L, e, role, src=None):
        offs, fd = self._layer(L)
        og, ou, od, g, dd = offs[e]
        o, n = {"gate": (og, g), "up": (ou, g), "down": (od, dd)}[role]
        return _pread(fd, n, o)


class ExpertRTN:
    """Round-to-nearest from the FP4 source at a fixed tier or a v41_alloc map."""

    def __init__(self, tier, amap=None, NE=384):
        self.tier, self.amap, self.NE = tier, amap, NE

    def ident(self):
        return "rtn:" + self.tier + ":" + hashlib.sha256(json.dumps(self.amap, sort_keys=True).encode()).hexdigest()

    def tiers(self, L):
        gu, dn = [self.tier] * self.NE, [self.tier] * self.NE
        m = (self.amap or {}).get(str(L))
        if m:
            for e, pr in m.items():
                e = int(e)
                for k, t in pr.items():
                    if k in ("all", "gu"):
                        gu[e] = t
                    if k in ("all", "dn"):
                        dn[e] = t
        return gu, dn

    def types(self, L):
        gu, dn = self.tiers(L)
        return [F.TIER_TYPE[t] for t in gu], [F.TIER_TYPE[t] for t in dn]

    def get(self, L, e, role, src):
        gu, dn = self.tiers(L)
        w = dict(ROLES)[role]
        W = src.f32(f"layers.{L}.ffn.experts.{e}.{w}.weight")
        return F.encode_rtn(W, dn[e] if role == "down" else gu[e]).tobytes()


class ExpertHoles(ExpertRTN):
    """--experts-in-place: the tiers of --rtn / --rtn-map, but no payload - each expert region is left as a hole for
    v41_quant.py --gguf to fill at the offsets in <shard>.experts.json."""

    def ident(self):
        return "inplace:" + super().ident()

    def get(self, L, e, role, src):
        raise RuntimeError("in-place experts have no payload here")


def experts_json(path, hdr_len, plan):
    """<shard>.experts.json: per layer the types and each expert's absolute (gate, up, down) byte offsets."""
    out = {}
    for t in plan:
        if t.kind != "experts":
            continue
        lay = out.setdefault(str(t.layer), {"gu_type": [0] * 384, "d_type": [0] * 384, "off": {}})
        per = t.nbytes // len(t.experts)
        for k, e in enumerate(t.experts):
            lay["d_type" if t.role == "down" else "gu_type"][e] = t.qtype
            lay["off"].setdefault(str(e), {})[t.role] = hdr_len + t.offset + k * per
    tmp = path + ".experts.json.tmp"
    json.dump({"file": os.path.basename(path), "layers": out}, open(tmp, "w"))
    os.replace(tmp, path + ".experts.json")


_W = {}


def _worker_init(provider):
    _W["p"], _W["src"] = provider, Source()


def _worker_get(args):
    L, e, role = args
    return _W["p"].get(L, e, role, _W["src"])


# ================================================================== encoding
def encode_regular(src, t, chunk=4096):
    """Yield the bytes of a regular tensor, in row chunks (bounded RAM for the 129280 x 5120 head)."""
    _, shape, _, _ = src.info(t.src)
    if len(shape) == 1:
        yield _enc(src.f32(t.src).reshape(1, -1), t.qtype, t.name)
        return
    for r0 in range(0, shape[0], chunk):   # chunk is a multiple of the FP8 32-row scale block
        yield _enc(src.f32(t.src, r0, r0 + chunk), t.qtype, t.name)


def _enc(v, qt, name):
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name}: nonfinite value")
    if qt == F.F32:
        return v.astype(np.float32).tobytes()
    if qt == F.F16:
        if np.any(np.abs(v) > 65504):
            raise ValueError(f"{name}: F16 overflow")
        return v.astype(np.float16).tobytes()
    if qt == F.Q8_0:
        return F.encode_q8_0(v).tobytes()
    raise ValueError(qt)


def engram_chunks(src, t, rows_per=16384):
    sn = t.src.removesuffix(".weight") + ".scale"
    rows = t.shape[1]
    for r0 in range(0, rows, rows_per):
        n = min(rows_per, rows - r0)
        w = np.frombuffer(src.read(t.src, r0 * 256, n * 256), np.uint8).reshape(n, 256)
        s = np.frombuffer(src.read(sn, r0 * 8, n * 8), np.uint8).reshape(n, 8)
        if np.any((w & 127) == 127) or np.any(s == 255):
            raise ValueError(f"{t.name}: nonfinite Engram row near {r0}")
        p = np.empty((n, ENGRAM_ROW), np.uint8)
        p[:, :256], p[:, 256:] = w, s
        yield p.tobytes()


# ================================================================== file layout
def tensor_info(t):
    return MD.pack_string(t.name) + struct.pack("<I", len(t.shape)) + struct.pack(f"<{len(t.shape)}Q", *t.shape) + \
        struct.pack("<IQ", t.qtype, t.offset)


def header(tensors, records):
    h = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(records)) + b"".join(records) + \
        b"".join(tensor_info(t) for t in tensors)
    return h + bytes(align(len(h)) - len(h))


def shard_paths(out, engram_dir=None):
    stem = out[:-5] if out.endswith(".gguf") else out
    p1, p2 = f"{stem}-00001-of-00002.gguf", f"{stem}-00002-of-00002.gguf"
    real2 = os.path.join(engram_dir, os.path.basename(p2)) if engram_dir else p2
    return p1, p2, real2


def split_records(no, n_tensors_total):
    return [MD.kv_u16("split.no", no), MD.kv_u16("split.count", 2), MD.kv_i32("split.tensors.count", n_tensors_total)]


def write_shard(path, tensors, records, src, provider, threads, ident):
    hdr = header(tensors, records)
    sig = hashlib.sha256(hdr + ident.encode()).hexdigest()
    partial, journal = path + ".partial", path + ".partial.json"
    if os.path.exists(path):
        raise ValueError(f"refusing to overwrite {path}")
    done = 0
    if os.path.exists(journal):
        j = json.load(open(journal))
        if j["signature"] != sig:
            raise ValueError(f"{journal}: a different plan/source wrote this partial file; delete it to restart")
        done = j["completed"]
        log(f"{os.path.basename(path)}: resuming after {done}/{len(tensors)} tensors")
    total = len(hdr) + (tensors[-1].offset + align(tensors[-1].nbytes) if tensors else 0)
    end = len(hdr) + (tensors[done - 1].offset + align(tensors[done - 1].nbytes) if done else 0)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
    if free < total - end + (8 << 30):
        raise ValueError(f"{path}: needs {(total - end) / 2**30:.1f} GiB + 8 GiB reserve, {free / 2**30:.1f} GiB free")
    mode = "r+b" if done else "wb"
    if not done and os.path.exists(partial):
        os.unlink(partial)
    pool = None
    with open(partial, mode) as fp:
        if not done:
            fp.write(hdr)
            json.dump({"signature": sig, "completed": 0}, open(journal, "w"))
        fp.truncate(end)
        fp.seek(end)
        try:
            for i in range(done, len(tensors)):
                t, t0, n = tensors[i], time.time(), 0
                if fp.tell() != len(hdr) + t.offset:
                    raise ValueError(f"{t.name}: at {fp.tell()}, planned {len(hdr) + t.offset}")
                if t.kind == "regular":
                    parts = encode_regular(src, t)
                elif t.kind == "etype":
                    parts = [np.asarray(t.experts, np.int32).tobytes()]
                elif t.kind == "engram":
                    parts = engram_chunks(src, t)
                elif isinstance(provider, ExpertHoles):   # filled later by v41_quant.py --gguf
                    fp.seek(t.nbytes, 1)
                    parts, n = [], t.nbytes
                else:
                    jobs = [(t.layer, e, t.role) for e in t.experts]
                    if isinstance(provider, ExpertRTN) and threads > 1:
                        if pool is None:
                            pool = cf.ProcessPoolExecutor(threads, initializer=_worker_init, initargs=(provider,))
                        parts = pool.map(_worker_get, jobs, chunksize=4)
                    else:
                        parts = (provider.get(L, e, r, src) for L, e, r in jobs)
                    per = t.nbytes // len(t.experts)
                for b in parts:
                    if t.kind == "experts" and len(b) != per:
                        raise ValueError(f"{t.name}: an expert encoded to {len(b)} bytes, expected {per}")
                    fp.write(b)
                    n += len(b)
                if n != t.nbytes:
                    raise ValueError(f"{t.name}: wrote {n} bytes, planned {t.nbytes}")
                fp.write(bytes(align(t.nbytes) - t.nbytes))
                fp.flush()
                os.fsync(fp.fileno())
                json.dump({"signature": sig, "completed": i + 1}, open(journal + ".tmp", "w"))
                os.replace(journal + ".tmp", journal)
                if t.nbytes > (64 << 20) or i + 1 == len(tensors):
                    log(f"[{i + 1}/{len(tensors)}] {t.name} {F.TYPE_NAME[t.qtype]} {t.nbytes / 2**30:.3f} GiB "
                        f"{time.time() - t0:.1f}s")
        finally:
            if pool is not None:
                pool.shutdown(cancel_futures=True)
    if isinstance(provider, ExpertHoles):
        experts_json(path, len(hdr), tensors)
    os.rename(partial, path)
    os.unlink(journal)


# ================================================================== reading back (verify)
def read_gguf(path):
    """(kv {key: (type, raw value or list)}, [(name, shape, type, offset)], data_start)."""
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize(fmt)
            return struct.unpack(fmt, f.read(n))

        def rs():
            (n,) = rd("<Q")
            return f.read(n).decode("utf-8", "replace")

        def val(ty):
            if ty == MD.GGUF_STRING:
                return rs()
            if ty == MD.GGUF_ARRAY:
                et, n = rd("<IQ")
                return [val(et) for _ in range(n)] if et in (MD.GGUF_STRING, MD.GGUF_ARRAY) else \
                    list(rd("<" + str(n) + {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q",
                                            12: "d"}[et]))
            return rd("<" + {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}[ty])[0]

        assert f.read(4) == b"GGUF"
        ver, nt, nk = rd("<IQQ")
        assert ver == 3
        kv = {}
        for _ in range(nk):
            k = rs()
            (ty,) = rd("<I")
            kv[k] = val(ty)
        ts = []
        for _ in range(nt):
            name = rs()
            (nd,) = rd("<I")
            shape = rd(f"<{nd}Q")
            ty, off = rd("<IQ")
            ts.append((name, tuple(shape), ty, off))
        start = align(f.tell(), kv.get("general.alignment", 32))
    return kv, ts, start


def verify(p1, src, provider, cfg, layers, sample=3):
    kv, ts, start = read_gguf(p1)
    p2 = p1.replace("-00001-of-00002", "-00002-of-00002")
    have = {n: (s, t, o, start, p1) for n, s, t, o in ts}
    if p2 != p1 and os.path.exists(p2):
        kv2, ts2, start2 = read_gguf(p2)
        have.update({n: (s, t, o, start2, p2) for n, s, t, o in ts2})
    plan, disk, _ = build_plan(cfg, src, provider.types, layers, engram=True,
                               engram_rows=kv.get("strata.v41.engram_rows_limit") or None)
    want = {t.name: t for t in plan + (disk if p2 != p1 and os.path.exists(p2) else [])}
    missing, extra = set(want) - set(have), set(have) - set(want)
    if missing or extra:
        raise ValueError(f"tensor set differs: missing {sorted(missing)[:5]} extra {sorted(extra)[:5]}")
    bad, checked = [], 0
    rng = np.random.default_rng(41)
    for name, t in want.items():
        s, ty, off, base, path = have[name]
        if tuple(s) != tuple(t.shape) or ty != t.qtype:
            bad.append(f"{name}: {s} type {ty} vs plan {t.shape} {t.qtype}")
            continue
        with open(path, "rb") as f:
            if t.kind == "regular":
                f.seek(base + off)
                got = f.read(t.nbytes)
                if got != b"".join(encode_regular(src, t)):
                    bad.append(f"{name}: payload differs from the source recipe")
            elif t.kind == "etype":
                f.seek(base + off)
                if list(np.frombuffer(f.read(t.nbytes), np.int32)) != list(t.experts):
                    bad.append(f"{name}: type map differs")
            elif t.kind == "experts" and isinstance(provider, ExpertHoles):
                pass   # payload written by the quantize pass; tests/check_v41_gguf.py --fp4 checks it
            elif t.kind == "experts":
                per = t.nbytes // len(t.experts)
                for k in sorted({0, len(t.experts) - 1, int(rng.integers(len(t.experts)))})[:sample]:
                    f.seek(base + off + k * per)
                    if f.read(per) != provider.get(t.layer, t.experts[k], t.role, src):
                        bad.append(f"{name}: expert {t.experts[k]} (slot {k}) differs")
            else:
                rows = t.shape[1]
                sn = t.src.removesuffix(".weight") + ".scale"
                for r in sorted({0, 1, rows - 1, 16383 % rows, 16384 % rows, *rng.integers(rows, size=64).tolist()}):
                    f.seek(base + off + r * ENGRAM_ROW)
                    if f.read(ENGRAM_ROW) != src.read(t.src, r * 256, 256) + src.read(sn, r * 8, 8):
                        bad.append(f"{name}: Engram row {r} differs")
                        break
        checked += 1
    return checked, bad


# ================================================================== main
def parse_layers(s, n):
    if not s:
        return list(range(n))
    a, b = map(int, s.split("-"))
    return list(range(a, b + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--verify")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--experts")
    g.add_argument("--rtn", choices=list(F.TIER_TYPE))
    ap.add_argument("--rtn-map")
    ap.add_argument("--experts-in-place", action="store_true",
                    help="leave the expert regions as holes (tiers from --rtn / --rtn-map) for v41_quant.py --gguf")
    ap.add_argument("--layers")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--engram-rows", type=int)
    ap.add_argument("--engram-dir")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = Source()
    cfg = json.load(open(os.path.join(MAIN, "config.json")))
    c = cfg["text_config"]
    layers = parse_layers(a.layers, c["num_hidden_layers"])
    if a.experts:
        provider = ExpertDir(a.experts, c["hidden_size"], c["moe_intermediate_size"])
    else:
        amap = json.load(open(a.rtn_map))["map"] if a.rtn_map else None
        provider = (ExpertHoles if a.experts_in_place else ExpertRTN)(a.rtn or "q2_0", amap, c["n_routed_experts"])
    if a.verify:
        n, bad = verify(a.verify, src, provider, cfg, layers)
        print(f"verify: {n} tensors checked, {len(bad)} problems")
        for b in bad[:20]:
            print("  " + b)
        sys.exit(1 if bad else 0)
    if not a.out:
        ap.error("--out or --verify")
    plan, disk, claimed = build_plan(cfg, src, provider.types, layers, not a.no_engram, a.engram_rows)
    if a.layers is None:
        omitted = {n for n in src.wm if n.startswith(("mtp.", "vision.", "aligner.", "image_"))}
        rest = set(src.wm) - claimed - omitted
        if rest:
            raise ValueError(f"unclaimed source tensors: {sorted(rest)[:10]}")
    _, records = MD.metadata(MAIN, NAME, URL)
    counts = {}
    for t in plan:
        if t.kind == "experts" and t.role != "up":
            k = ("gu" if t.role == "gate" else "down", F.TYPE_NAME[t.qtype])
            counts[k] = counts.get(k, 0) + len(t.experts)
    records += [
        MD.kv_string("deepseek41.quantization", "routed experts: per-expert Q3_K / Q2_0 / PTQ1_0 (gate+up share a type, "
                     "down its own); Q8_0 attention / shared / head; F16 embedding"),
        MD.kv_string("strata.v41.expert_source", provider.ident()),
        MD.kv_string("strata.v41.expert_tiers", json.dumps({f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())})),
        MD.kv_string("strata.v41.expert_layout", "uniform role: ffn_<role>_exps.weight; mixed: ffn_<role>_exps.<type>.weight "
                     "per type (experts in id order) + I32 ffn_exps_{gu,down}_type[384]"),
    ]
    if a.layers is not None:
        records.append(MD.kv_u32_array("strata.v41.partial_layers", layers))
    if a.engram_rows:
        records.append(MD.kv_u32("strata.v41.engram_rows_limit", a.engram_rows))
    p1, p2, real2 = shard_paths(a.out, a.engram_dir)
    ntot = len(plan) + len(disk)
    if disk:
        r1, r2 = records + split_records(0, ntot), [MD.kv_string("general.architecture", "deepseek41"),
                                                     MD.kv_u32("general.alignment", ALIGN)] + split_records(1, ntot)
    else:   # no Engram (--no-engram, or --layers without 1/14): one plain file
        p1, r1, r2 = a.out, records, []
    size1 = len(header(plan, r1)) + sum(align(t.nbytes) for t in plan)
    size2 = len(header(disk, r2)) + sum(align(t.nbytes) for t in disk) if disk else 0
    log(f"plan: {len(plan)} + {len(disk)} tensors; shard 1 {size1 / 2**30:.2f} GiB, shard 2 (Engram) {size2 / 2**30:.2f} GiB; "
        f"experts {dict((f'{k[0]}:{k[1]}', v) for k, v in sorted(counts.items()))}")
    if a.dry_run:
        for t in plan + disk:
            print(f"{t.shard} {t.offset:>14} {t.nbytes:>13} {F.TYPE_NAME[t.qtype]:>6} {t.name} {t.shape}")
        return
    write_shard(p1, plan, r1, src, provider, a.threads, provider.ident())
    if disk:
        write_shard(real2, disk, r2, src, provider, a.threads, provider.ident())
        if real2 != p2:
            os.symlink(real2, p2)
    log(f"wrote {p1}" + (f" and {p2}" if disk else ""))


if __name__ == "__main__":
    main()
