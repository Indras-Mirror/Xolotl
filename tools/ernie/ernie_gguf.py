"""ernie_gguf.py - write an `ernie5_5` GGUF from the ERNIE-5.5-Flash FP8 safetensors, streaming and CPU-only.

    python3 tools/ernie/ernie_gguf.py SRC --layers 3 --experts q8_0 -o OUT/ernie-L3-q8.gguf
    python3 tools/ernie/ernie_gguf.py SRC -o OUT/ernie.gguf --experts mxfp4
    python3 tools/ernie/ernie_gguf.py SRC --check-source          # validate every present shard header
    python3 tools/ernie/ernie_gguf.py SRC --verify OUT/ernie-L3-q8.gguf --layers 3 --experts q8_0

SRC is the FP8 repo directory (config.json + model-NNNNN-of-NNNNN.safetensors).  Only COMPLETE shards are
indexed (a shard whose header or payload runs past EOF is skipped with a warning), so this can convert a
download in progress; `.cache/` partials are never touched and the source is never written.

Naming authority: llama.cpp PR #31925 (`conversion/ernie5.py`, `src/models/ernie5-5.cpp`) and
`include/strata/artifact/ernie_geometry.hpp`.  Tensor map, by role:

  token_embd / output / output_hc_*                 root
  blk.N.{attn_norm,ffn_norm,hc_attn_*,hc_ffn_*,attn_sinks,attn_vha,attn_q_a*,...}
  blk.N.{attn_kv,attn_compressor_*,attn_output_a/b}  HCA (compress ratio 128)
  blk.N.{attn_kv_a_mqa,attn_k_b,attn_v_b,indexer.*}  MLA + DSA (ratio 0, `-2` in config.json)
  blk.N.ffn_{gate,down,up}_exps                      routed experts, one stacked tensor per role
  blk.N.{ffn_routed_*,ffn_*_shexp,ffn_gate_inp,exp_*}  latent MoE

Types: every large non-expert matrix -> Q8_0; norms, APE, mHC mixers, the stacked router, sinks, the folded
VHA mix and the router bias -> F32 (what the PR's `tensor_force_quant` keeps); routed experts per `--experts`
(q8_0 = Q8_0 of the dequantised FP8, mxfp4 = OCP MXFP4, skip = no expert tensors).

`--layers N` writes blocks 0..N-1 and keeps the root tensors, like the DS4.1 converter's partial mode.
When a root tensor is not in the downloaded shards (lm_head, model.norm and the mHC head live in the last
shard), a truncated export writes zeros for it and says so in the log and in `ernie5_5.dev_placeholder_tensors`.

DSpark drafts (`model.mtp.*`, shard 128) go to a SEPARATE sidecar (`--dspark-out`), never the trunk file;
see `ernie_dspark_expected_tensors` for the naming.

Memory: one source chunk is at most a 128-row FP8 block (a whole 2048x2048 expert is 16 MB dequantised);
shards are mmap'd and the mappings are dropped between layers.  Peak RSS stays well under 1 GB.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "v41"))
import gguf_reader as GH  # noqa: E402  (the in-repo header reader, used by --verify)
import v41_formats as Q  # noqa: E402  (the public Q8_0 encoder)

GGUF_MAGIC = 0x46554747
ALIGN = 32
BLOCK = {"F32": (1, 4), "F16": (1, 2), "Q8_0": (32, 34), "MXFP4": (32, 17)}
TYPE_ID = {"F32": 0, "F16": 1, "Q8_0": 8, "MXFP4": 39, "BF16": 30}
META_ID = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6, "bool": 7, "string": 8,
           "array": 9, "u64": 10, "i64": 11, "f64": 12}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def align(n, a=ALIGN):
    return (n + a - 1) // a * a


# ===================================================================================== safetensors source
class SrcTensor:
    __slots__ = ("name", "path", "offset", "dtype", "shape", "nbytes")

    def __init__(self, name, path, offset, dtype, shape, nbytes):
        self.name, self.path, self.offset, self.dtype, self.shape, self.nbytes = (
            name, path, offset, dtype, shape, nbytes)


class Store:
    """The complete-shard index plus a bounded cache of mmaps.  Never loads a full shard."""

    DTYPE_BYTES = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4, "I8": 1, "I16": 2, "I32": 4, "I64": 8}

    def __init__(self, src_dir, max_open=4):
        self.dir = src_dir
        self.index: dict[str, SrcTensor] = {}
        self.files: list[str] = []
        self.skipped: list[tuple[str, str]] = []
        self._maps: dict[str, tuple[object, object]] = {}
        self._fds: dict[str, int] = {}
        import threading
        self._fd_lock = threading.Lock()
        self._max_open = max_open
        self.per_file = {}
        self.io = {}   # basename -> [bytes, seconds] (mapped-page copies, i.e. the real read cost)
        self._scan()

    def _scan(self):
        names = sorted(n for n in os.listdir(self.dir)
                       if n.endswith(".safetensors") and n.startswith("model-"))
        if not names:
            raise SystemExit(f"{self.dir}: no model-*.safetensors shards")
        for n in names:
            p = os.path.join(self.dir, n)
            try:
                self._index_file(p)
                self.files.append(p)
            except ValueError as e:
                self.skipped.append((p, str(e)))
                log(f"skip {n}: {e}")

    def _index_file(self, path):
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            if hlen <= 0 or 8 + hlen > size:
                raise ValueError(f"header length {hlen} past EOF ({size} B)")
            hdr = json.loads(f.read(hlen))
        end = 8 + hlen
        n = 0
        local = {}
        for name, t in hdr.items():
            if name == "__metadata__":
                continue
            dt, shape, off = t["dtype"], t["shape"], t["data_offsets"]
            if dt not in self.DTYPE_BYTES:
                raise ValueError(f"{name}: dtype {dt} is not supported")
            nbytes = off[1] - off[0]
            row = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
            if nbytes != row * shape[0] * self.DTYPE_BYTES[dt] and len(shape) > 0:
                raise ValueError(f"{name}: payload {nbytes} B does not match {dt}{shape}")
            end = max(end, 8 + hlen + off[1])
            if name in self.index or name in local:
                raise ValueError(f"duplicate tensor {name} in {path}")
            local[name] = SrcTensor(name, path, 8 + hlen + off[0], dt, tuple(shape), nbytes)
            n += 1
        if end > size:
            raise ValueError(f"payload ends at {end} B, file is {size} B (still downloading)")
        self.index.update(local)
        self.per_file[path] = n

    # ---- reading
    def _map(self, path):
        hit = self._maps.get(path)
        if hit is not None:
            self._maps[path] = hit  # refresh LRU order
            return hit
        f = open(path, "rb")
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self._maps) >= self._max_open:
            old = next(iter(self._maps))
            of, om = self._maps.pop(old)
            om.close()
            of.close()
        self._maps[path] = (f, m)
        return f, m

    def _fd(self, path):
        """A read-only fd per shard, shared by threads (pread is positional, so sharing is safe); closed in close()."""
        fd = self._fds.get(path)
        if fd is None:
            with self._fd_lock:
                fd = self._fds.get(path)
                if fd is None:
                    fd = os.open(path, os.O_RDONLY)
                    self._fds[path] = fd
        return fd

    def raw(self, name, r0=0, r1=None):
        t = self.index[name]
        fd = self._fd(t.path)
        rows = t.shape[0] if t.shape else 1
        r1 = rows if r1 is None else r1
        row_bytes = t.nbytes // rows
        a = t.offset + r0 * row_bytes
        b = t.offset + r1 * row_bytes
        t0 = time.time()
        # One positioned read, not an mmap slice: page faults arrive as ~85 KB readahead requests, which on a
        # rotational source drive that is also being written to turns into seek-bound I/O (99 % util at <1 MB/s read).
        parts, pos = [], a
        while pos < b:
            chunk = os.pread(fd, min(b - pos, 1 << 30), pos)
            if not chunk:
                raise IOError(f"{t.path}: short read at {pos} (wanted {b - pos} more bytes)")
            parts.append(chunk)
            pos += len(chunk)
        buf = parts[0] if len(parts) == 1 else b"".join(parts)
        io = self.io.setdefault(os.path.basename(t.path), [0, 0.0])
        io[0] += b - a
        io[1] += time.time() - t0
        return buf

    def f32(self, name, r0=0, r1=None, scale_name=None):
        """Rows [r0, r1) dequantised to float32.  FP8 needs r0 to be a multiple of 128 (the block grid)."""
        t = self.index[name]
        rows = t.shape[0] if t.shape else 1
        r1 = rows if r1 is None else r1
        cols = int(np.prod(t.shape[1:], dtype=np.int64)) if len(t.shape) > 1 else 1
        buf = self.raw(name, r0, r1)
        n = r1 - r0
        if t.dtype == "F8_E4M3":
            if r0 % 128:
                raise ValueError(f"{name}: FP8 reads must start on a 128-row block ({r0})")
            w = E4M3[np.frombuffer(buf, np.uint8)].reshape(n, cols)
            sn = scale_name or (t.name.removesuffix(".weight") + ".weight_scale_inv")
            if sn not in self.index:
                sn = t.name + "_scale_inv"
            st = self.index[sn]
            # scales are [ceil(R/128), ceil(C/128)]; the chunk is whole blocks
            sbuf = self.raw(sn, r0 // 128, (r1 + 127) // 128)
            s = np.frombuffer(sbuf, "<f4").reshape(-1, st.shape[1]).astype(np.float32)
            check_ue8m0(s, sn)
            s = np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)[:n, :cols]
            return w * s
        if t.dtype == "BF16":
            u = np.frombuffer(buf, "<u2").astype(np.uint32) << 16
            return u.view(np.float32).reshape(n, cols)
        if t.dtype == "F32":
            return np.frombuffer(buf, "<f4").astype(np.float32).reshape(n, cols)
        if t.dtype == "F16":
            return np.frombuffer(buf, "<f2").astype(np.float32).reshape(n, cols)
        raise ValueError(f"{name}: cannot dequantise {t.dtype}")

    def report_io(self):
        for name in sorted(self.io):
            nb, dt = self.io[name]
            log(f"  read {name}: {nb / 2**30:6.2f} GiB in {dt:7.1f}s "
                f"({nb / 2**20 / max(dt, 1e-9):6.1f} MiB/s)")

    def drop(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        for of, om in self._maps.values():
            om.close()
            of.close()
        self._maps.clear()


def _e4m3_lut():
    v = np.zeros(256, np.float32)
    for b in range(256):
        e, mant = (b >> 3) & 0xF, b & 0x7
        if e == 0xF and mant == 0x7:
            v[b] = np.nan
        elif e == 0:
            v[b] = mant * 2.0 ** -9
        else:
            v[b] = (1.0 + mant / 8.0) * 2.0 ** (e - 7)
        if b & 0x80:
            v[b] = -v[b]
    return v


E4M3 = _e4m3_lut()
_WARNED_SCALES = set()


def check_ue8m0(s, name):
    """ue8m0 stores a power of two in an F32, so every mantissa bit must be zero (a non-zero mantissa is the
    documented fallback: use the float as it is, but say so once)."""
    bits = np.asarray(s, np.float32).view(np.uint32)
    bad = (bits & np.uint32(0x007FFFFF)) != 0
    if np.any(bad) and name not in _WARNED_SCALES:
        _WARNED_SCALES.add(name)
        log(f"warning: {name} holds a non-power-of-two scale; using it as a float")


# ===================================================================================== encoders
KVALUES = np.array((0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), np.int8)


def e8m0_to_f32(x):
    x = np.asarray(x, np.uint32)
    bits = np.where(x < 2, np.uint32(0x00200000) << x, (x - 1) << np.uint32(23))
    return bits.view(np.float32) if bits.flags.c_contiguous else bits.astype(np.uint32).view(np.float32)


def encode_mxfp4(W):
    """ggml `quantize_row_mxfp4_ref`: [R, C] f32 (C % 32 == 0) -> [R, C/32*17] bytes.
    One E8M0 byte per 32 values, then 16 bytes: qs[j] = elem j | elem j+16 << 4.
    The shared exponent is floor(log2(amax)) - 2 and the e2m1 table is doubled, exactly as ggml does it."""
    W = np.ascontiguousarray(W, np.float32)
    R, C = W.shape
    if C % 32:
        raise ValueError(f"MXFP4 row length {C} is not a multiple of 32")
    g = W.reshape(R, C // 32, 32)
    amax = np.abs(g).max(-1, keepdims=True)
    with np.errstate(divide="ignore"):
        e = np.where(amax > 0, np.floor(np.log2(amax)) - 2 + 127, 0).astype(np.uint8)
    d = e8m0_to_f32(e).astype(np.float32)
    # Nearest doubled-e2m1 value.  The 8 positive magnitudes are 0,1,2,3,4,6,8,12, so the decision bounds are
    # the midpoints 0.5,1.5,2.5,3.5,5,7,10 (times d).  A tie keeps the lower index, as ggml's best_index_mxfp4
    # does with its strict `<`; negatives reuse the same search (index 0 is the shared zero).
    m = np.abs(g)
    j = np.zeros_like(m, np.uint8)
    for t in (0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0):
        j += (m > np.float32(t) * d).astype(np.uint8)
    best = np.where(g < 0, np.where(j == 0, 0, 8 + j), j).astype(np.uint8)
    out = np.empty((R, C // 32, 17), np.uint8)
    out[..., 0] = e[..., 0]
    out[..., 1:] = best[..., :16] | (best[..., 16:] << 4)
    return out.reshape(R, -1)


def encode(W, gtype):
    if gtype == "F32":
        return np.ascontiguousarray(W, "<f4").tobytes()
    if gtype == "F16":
        return np.ascontiguousarray(W, "<f2").tobytes()
    if gtype == "Q8_0":
        return Q.encode_q8_0(np.asarray(W, np.float32)).tobytes()
    if gtype == "MXFP4":
        return encode_mxfp4(W).tobytes()
    raise ValueError(gtype)


def row_chunks(rows, dim0, gtype, chunk_rows=None):
    """Chunk size that stays well under 128 MB for the widest tensor (rows are dim0 long)."""
    if gtype in ("F32", "F16"):
        per = dim0 * BLOCK[gtype][1]
    else:
        per = dim0 // BLOCK[gtype][0] * BLOCK[gtype][1]
    cap = max(1, (32 << 20) // max(1, per))
    return min(rows, chunk_rows or cap)


def nbytes_for(shape, gtype):
    total = int(np.prod(shape))
    if gtype in ("F32", "F16"):
        return total * BLOCK[gtype][1]
    if total % BLOCK[gtype][0]:
        raise ValueError(f"shape {shape} is not a whole number of {gtype} blocks")
    return total // BLOCK[gtype][0] * BLOCK[gtype][1]


# ===================================================================================== plan
class Out:
    """One output tensor: declared GGUF shape, type, byte count and a chunk generator."""

    def __init__(self, name, shape, gtype, emit, sample=None):
        self.name = name
        self.shape = list(shape)
        self.gtype = gtype
        self.nbytes = nbytes_for(shape, gtype)
        self.emit = emit
        self.sample = sample  # (src_name, rows_per_expert or None) for --verify


class Plan:
    def __init__(self, store, opts):
        self.s = store
        self.o = opts
        self.out: list[Out] = []
        self.placeholders: list[str] = []
        self.missing: list[str] = []
        self.saw: set[str] = set()
        self.check_only = opts.check_source
        self.expert_names: dict[str, list[str]] = {}

    # ---- source presence
    def has(self, name):
        return name in self.s.index

    def need(self, name):
        self.saw.add(name)
        if not self.has(name):
            self.missing.append(name)
            return False
        t = self.s.index[name]
        if t.dtype == "F8_E4M3":
            sn = name.removesuffix(".weight") + ".weight_scale_inv"
            if sn not in self.s.index:
                sn = name + "_scale_inv"
            self.saw.add(sn)
            if sn not in self.s.index:
                self.missing.append(sn)
                return False
        return True

    def note_placeholder(self, gguf_name):
        self.placeholders.append(gguf_name)

    # ---- output helpers
    def _add(self, out):
        if not self.check_only:
            self.out.append(out)

    def small(self, sources, dst, shape, gtype, fn=None, sample=None):
        """Read every source whole (small tensors only), apply fn (list of f32 arrays) -> array."""
        if not all(self.need(s) for s in sources):
            return
        if self.check_only:
            return
        arrs = [self.s.f32(s) if self.s.index[s].dtype != "F32" else
                np.frombuffer(self.s.raw(s), "<f4").reshape(self.s.index[s].shape) for s in sources]
        if fn is None:
            a0 = arrs[0]
            data = a0.reshape(-1) if a0.ndim == 2 and a0.shape[1] == 1 else a0
        else:
            data = fn(arrs)
        raw = encode(np.asarray(data, np.float32), gtype)
        # a plain (untransformed) small tensor is directly comparable with its source in --verify
        if sample is None and fn is None:
            sample = (sources[0], None)
        self._add(Out(dst, shape, gtype, lambda: iter((raw,)), sample))

    def mat(self, src, dst, shape, gtype, sample=True):
        """A row-major matrix: source torch [rows, dim0] -> GGUF shape [dim0, rows, ...]."""
        if not self.need(src):
            return
        if self.check_only:
            return
        t = self.s.index[src]
        rows = int(np.prod(t.shape[:-1]))
        dim0 = t.shape[-1]
        if rows * dim0 != int(np.prod(shape)):
            raise ValueError(f"{dst}: source {t.shape} does not cover {shape}")
        fp8 = t.dtype == "F8_E4M3"

        def gen():
            step = 1024 if fp8 else row_chunks(rows, dim0, gtype)
            if fp8:
                step -= step % 128
            for r0 in range(0, rows, step):
                r1 = min(rows, r0 + step)
                yield encode(self.s.f32(src, r0, r1), gtype)

        self._add(Out(dst, shape, gtype, gen, (src, None) if sample else None))

    def stacked(self, src_names, dst, shape, gtype, per_rows, dim0):
        """n sources, each torch [per_rows, dim0] -> one tensor stacked on the slowest axis (experts)."""
        for n in src_names:
            if not self.need(n):
                return
        if self.check_only:
            return
        n = len(src_names)

        def one(name):
            t = self.s.index[name]
            if gtype in ("MXFP4", "Q8_0", "F16", "F32") and t.dtype in ("F8_E4M3", "BF16", "F16", "F32"):
                # row-independent encoders: one call over the whole expert == the concatenated row chunks
                return encode(self.s.f32(name, 0, per_rows), gtype)
            step = 128 if t.dtype == "F8_E4M3" else row_chunks(per_rows, dim0, gtype)
            return b"".join(encode(self.s.f32(name, r0, min(per_rows, r0 + step)), gtype)
                            for r0 in range(0, per_rows, step))

        def gen():
            nth = int(os.environ.get("ERNIE_GGUF_THREADS", "0")) or min(12, os.cpu_count() or 1)
            if nth <= 1:
                for name in src_names:
                    yield one(name)
                return
            from concurrent.futures import ThreadPoolExecutor
            window = 2 * nth   # bounded look-ahead: memory ~ window x one expert's f32 temporaries
            with ThreadPoolExecutor(nth) as ex:
                futs = [ex.submit(one, nm) for nm in src_names[:window]]
                nxt = len(futs)
                for k in range(len(src_names)):
                    b = futs[k].result()
                    futs[k] = None
                    if nxt < len(src_names):
                        futs.append(ex.submit(one, src_names[nxt]))
                        nxt += 1
                    yield b

        self._add(Out(dst, shape, gtype, gen, (src_names[0], per_rows)))

    def zeros(self, dst, shape, gtype):
        if self.check_only:
            return
        self.note_placeholder(dst)
        # a zero row repeats; emit it without materialising the tensor
        if gtype in ("F32", "F16"):
            row = b"\0" * (shape[0] * BLOCK[gtype][1])
            count = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        else:
            row = encode(np.zeros((1, shape[0]), np.float32), gtype)
            count = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        self._add(Out(dst, shape, gtype, lambda: iter([row] * count), None))

    def maybe_small_or_zero(self, src, dst, shape, gtype, fn=None):
        if not self.has(src) and self.o.layers:
            self.zeros(dst, shape, gtype)
            return
        self.small([src], dst, shape, gtype, fn)


# ===================================================================================== model plan
def read_config(src):
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    if "multimax_ranges" in cfg and "multimax_ts" in cfg:
        cfg["_multimax"] = list(cfg["multimax_ranges"]) + list(cfg["multimax_ts"])
    return cfg


def plan_model(pl: Plan, cfg, n_export):
    """Root + trunk blocks 0..n_export-1."""
    D = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    dh = cfg["head_dim"]
    rq = cfg["q_lora_rank"]
    og, ol = cfg["o_groups"], cfg["o_lora_rank"]
    dk = cfg["hybrid_mla_qk_nope_head_dim"] + cfg["hybrid_mla_qk_rope_head_dim"]
    dv = cfg["hybrid_mla_v_head_dim"]
    kl = cfg["kv_lora_rank"]
    qk_nope = cfg["hybrid_mla_qk_nope_head_dim"]
    nff = cfg["intermediate_size"]
    ne = cfg["n_routed_experts"]
    fe = cfg["moe_intermediate_size"]
    lat = cfg["moe_latent_size"]
    V = cfg["vocab_size"]
    nse = cfg["n_shared_experts"]
    ratios = [max(r, 0) for r in cfg["compress_ratios"]]
    dense_lead = cfg["first_k_dense_replace"]
    exp_type = {"q8_0": "Q8_0", "mxfp4": "MXFP4", "f16": "F16", "f32": "F32", "skip": None}[pl.o.experts]
    bt = {"q8_0": "Q8_0", "f16": "F16", "f32": "F32"}[pl.o.ftype]

    # The Preview's own MultiMax (ranges then ts, both BF16 [4]) lives in the last shard and replaces the Base
    # placeholder values that --metadata-from used to copy.
    multimax = []
    for leaf in ("multimax_ranges", "multimax_ts"):
        name = f"model.lm_head.{leaf}"
        if pl.has(name):
            pl.saw.add(name)
            if not pl.check_only:
                multimax.append(pl.s.f32(name).reshape(-1))
    if len(multimax) == 2:
        cfg["_multimax"] = [float(x) for x in np.concatenate(multimax)]

    # ---- root
    pl.mat("model.embed_tokens.weight", "token_embd.weight", [D, V], bt)
    pl.mat("lm_head.weight", "output.weight", [D, V], bt)
    if not pl.has("lm_head.weight") and pl.o.layers:
        pl.zeros("output.weight", [D, V], bt)
    pl.maybe_small_or_zero("model.norm.weight", "output_norm.weight", [D], "F32")
    for leaf, dst, shape, fn in (
            ("mhc_contract.hc_head_fn", "output_hc_fn.weight", [D * cfg["hc_mult"], cfg["hc_mult"]], lambda a: a[0].T.copy()),
            ("mhc_contract.hc_head_base", "output_hc_base.weight", [cfg["hc_mult"]], lambda a: a[0].reshape(-1)),
            ("mhc_contract.hc_head_scale", "output_hc_scale.weight", [1], lambda a: a[0].reshape(-1))):
        pl.maybe_small_or_zero(f"model.{leaf}", dst, shape, "F32", fn)

    mhc = cfg["hc_mult"]
    hcd, hcm = mhc * D, (2 + mhc) * mhc

    def hc_pair(bid, kind, src_kind):
        base = f"model.layers.{bid}.{src_kind}_hyper_connection."
        dst = f"blk.{bid}.hc_{kind}_"
        pl.maybe_small_or_zero(base + "mapping_proj.weight", dst + "fn.weight", [hcd, hcm], "F32",
                               lambda a: a[0].T.copy())
        pl.maybe_small_or_zero(base + "bias", dst + "base.weight", [hcm], "F32")
        pl.small([base + "alpha_pre", base + "alpha_post", base + "alpha_res"], dst + "scale.weight", [3], "F32",
                 lambda a: np.concatenate([x.reshape(1) for x in a]))

    for bid in range(n_export):
        p = f"model.layers.{bid}."
        g = f"blk.{bid}."
        is_hca = ratios[bid] != 0

        # shared per-block
        pl.maybe_small_or_zero(p + "input_layernorm.weight", g + "attn_norm.weight", [D], "F32")
        pl.maybe_small_or_zero(p + "post_attention_layernorm.weight", g + "ffn_norm.weight", [D], "F32")
        hc_pair(bid, "attn", "self_attention")
        hc_pair(bid, "ffn", "mlp")
        # the sink and the q-latent have different HF names on the two layer kinds
        sink = p + ("self_attn.core_attention.attn_sink" if is_hca
                    else "self_attn.core_attention.softmax_offset")
        pl.maybe_small_or_zero(sink, g + "attn_sinks.weight", [nh], "F32")
        pl.small([p + "self_attn.vha_postmix_U", p + "self_attn.vha_postmix_V"], g + "attn_vha.weight", [nh, nh],
                 "F32", fold_vha)
        pl.maybe_small_or_zero(p + ("self_attn.linear_q_down_proj.weight" if is_hca
                                    else "self_attn.q_a_proj.weight"), g + "attn_q_a.weight", [D, rq], bt)
        pl.maybe_small_or_zero(p + ("self_attn.q_layernorm.weight" if is_hca
                                    else "self_attn.q_a_layernorm.weight"), g + "attn_q_a_norm.weight", [rq], "F32")

        if is_hca:
            pl.mat(p + "self_attn.linear_q_up_proj.weight", g + "attn_q_b.weight", [rq, nh * dh], bt)
            pl.mat(p + "self_attn.linear_kv_proj.weight", g + "attn_kv.weight", [D, dh], bt)
            pl.maybe_small_or_zero(p + "self_attn.kv_layernorm.weight", g + "attn_kv_a_norm.weight", [dh], "F32")
            pl.mat(p + "self_attn.linear_o_group_proj", g + "attn_output_a.weight",
                   [nh * dh // og, ol, og], bt)
            pl.mat(p + "self_attn.o_proj.weight", g + "attn_output_b.weight", [og * ol, D], bt)
            pl.mat(p + "self_attn.gate_proj.weight", g + "attn_gate.weight", [rq, og * ol], bt)
            pl.mat(p + "self_attn.core_attention.compressor.linear_wkv.weight",
                   g + "attn_compressor_kv.weight", [D, dh], bt)
            pl.mat(p + "self_attn.core_attention.compressor.linear_wgate.weight",
                   g + "attn_compressor_gate.weight", [D, dh], bt)
            pl.small([p + "self_attn.core_attention.compressor.ape"], g + "attn_compressor_ape.weight",
                     [dh, ratios[bid]], "F32")
            pl.maybe_small_or_zero(p + "self_attn.core_attention.compressor.norm.weight",
                                   g + "attn_compressor_norm.weight", [dh], "F32")
        else:
            pl.mat(p + "self_attn.q_b_proj.weight", g + "attn_q_b.weight", [rq, nh * dk], bt)
            pl.mat(p + "self_attn.kv_a_proj_with_mqa.weight", g + "attn_kv_a_mqa.weight", [D, kl + cfg["qk_rope_head_dim"]], bt)
            pl.maybe_small_or_zero(p + "self_attn.kv_a_layernorm.weight", g + "attn_kv_a_norm.weight", [kl], "F32")
            pl.split_kv_b(p + "self_attn.kv_b_proj.weight", g, qk_nope, dv, kl, nh)
            pl.mat(p + "self_attn.o_proj.weight", g + "attn_output.weight", [nh * dv, D], bt)
            pl.mat(p + "self_attn.gate_proj.weight", g + "attn_gate.weight", [rq, nh * dv], bt)
            pl.mat(p + "self_attn.core_attention.indexer.wk.weight", g + "indexer.attn_k.weight", [D, cfg["index_head_dim"]], bt)
            pl.small([p + "self_attn.core_attention.indexer.k_norm.weight"], g + "indexer.k_norm.weight",
                     [cfg["index_head_dim"]], "F32")
            pl.small([p + "self_attn.core_attention.indexer.k_norm.bias"], g + "indexer.k_norm.bias",
                     [cfg["index_head_dim"]], "F32")
            pl.mat(p + "self_attn.core_attention.indexer.wq_b.weight", g + "indexer.attn_q_b.weight",
                   [rq, cfg["index_n_heads"] * cfg["index_head_dim"]], bt)
            pl.small([p + "self_attn.core_attention.indexer.weights_proj.weight"], g + "indexer.proj.weight",
                     [D, cfg["index_n_heads"]], "F32")

        if bid < dense_lead:
            pl.mat(p + "mlp.gate_proj.weight", g + "ffn_gate.weight", [D, nff], bt)
            pl.mat(p + "mlp.up_proj.weight", g + "ffn_up.weight", [D, nff], bt)
            pl.mat(p + "mlp.down_proj.weight", g + "ffn_down.weight", [nff, D], bt)
            continue

        # latent MoE
        pl.small([p + "block_sparse_moe.gate.weight", p + "block_sparse_moe.gate.weight_1"],
                 g + "ffn_gate_inp.weight", [D, 2 * ne], "F32",
                 lambda a: np.concatenate(a, axis=0))
        esc = p + "block_sparse_moe.e_score_correction_bias"
        if not pl.has(esc):
            esc = p + "block_sparse_moe.e_score_correction.bias"
        pl.maybe_small_or_zero(esc, g + "exp_probs_b.bias", [ne], "F32")
        pl.maybe_small_or_zero(p + "block_sparse_moe.gate.routed_scaling_factor_param", g + "exp_scale.weight",
                               [ne], "F32")
        pl.mat(p + "block_sparse_moe.fc1_latent_proj.weight", g + "ffn_routed_down.weight", [D, lat], bt)
        pl.mat(p + "block_sparse_moe.fc2_latent_proj.weight", g + "ffn_routed_up.weight", [lat, D], bt)
        pl.maybe_small_or_zero(p + "block_sparse_moe.latent_norm.weight", g + "ffn_routed_norm.weight", [lat], "F32")
        pl.mat(p + "block_sparse_moe.shared_experts.w1.weight", g + "ffn_gate_shexp.weight", [D, fe * nse], bt)
        pl.mat(p + "block_sparse_moe.shared_experts.w2.weight", g + "ffn_down_shexp.weight", [fe * nse, D], bt)
        pl.mat(p + "block_sparse_moe.shared_experts.w3.weight", g + "ffn_up_shexp.weight", [D, fe * nse], bt)

        if exp_type is None:
            continue
        expect = [f"{p}block_sparse_moe.experts.{e}.w{r}.weight" for e in range(ne) for r in (1, 2, 3)]
        present = [n for n in expect if pl.has(n)]
        if pl.check_only:
            for n in expect:
                pl.need(n)
            continue
        if len(present) != len(expect):
            # A stacked expert tensor with fewer than n_expert slots would not match the architecture map.
            raise SystemExit(f"blk.{bid}: only {len(present)}/{len(expect)} routed-expert tensors are in the "
                             f"complete shards (that layer's shard is still downloading)")
        ne_done = ne
        for role, (src_role, dst_role, shape) in {
                "gate": ("w1", "ffn_gate_exps", [lat, fe, ne_done]),
                "down": ("w2", "ffn_down_exps", [fe, lat, ne_done]),
                "up": ("w3", "ffn_up_exps", [lat, fe, ne_done])}.items():
            names = [f"{p}block_sparse_moe.experts.{e}.{src_role}.weight" for e in range(ne_done)]
            pl.stacked(names, g + dst_role + ".weight", shape, exp_type, fe, lat)

    # exact set accounting for --check-source
    if not pl.check_only:
        if pl.missing and not pl.o.layers:
            raise SystemExit("missing source tensors:\n  " + "\n  ".join(pl.missing[:20]) +
                             (f"\n  ... {len(pl.missing)} total" if len(pl.missing) > 20 else ""))


def fold_vha(a):
    u, v = a[0].astype(np.float32), a[1].astype(np.float32)
    return (np.eye(u.shape[0], dtype=np.float32) + v @ u.T).astype(np.float32)


def _split_kv_b(self, src, g, qk_nope, dv, kl, nh):
    if not self.need(src):
        return
    if self.check_only:
        return
    bt = {"q8_0": "Q8_0", "f16": "F16", "f32": "F32"}[self.o.ftype]
    t = self.s.index[src]
    head_rows = qk_nope + dv
    # FP8 scale blocks are 128 rows, and a head is 448 rows, so only chunk sizes whose row span is a multiple
    # of 128 (lcm(448, 128) = 896 = 2 heads) may be read through Store.f32.
    step_heads = 1
    if t.dtype == "F8_E4M3":
        step_heads = 2
        if (step_heads * head_rows) % 128:
            raise ValueError(f"{src}: head stride {head_rows} cannot be read on the FP8 block grid")
    else:
        step_heads = max(1, row_chunks(nh, head_rows, bt))

    def gen_k():
        for h0 in range(0, nh, step_heads):
            h1 = min(nh, h0 + step_heads)
            full = self.s.f32(src, h0 * head_rows, h1 * head_rows)
            full = full.reshape(h1 - h0, head_rows, kl)
            kb = np.ascontiguousarray(full[:, :qk_nope, :].transpose(0, 2, 1))
            yield encode(kb.reshape(-1, qk_nope), bt)

    def gen_v():
        for h0 in range(0, nh, step_heads):
            h1 = min(nh, h0 + step_heads)
            full = self.s.f32(src, h0 * head_rows, h1 * head_rows)
            full = full.reshape(h1 - h0, head_rows, kl)
            vb = np.ascontiguousarray(full[:, qk_nope:, :])
            yield encode(vb.reshape(-1, kl), bt)

    self._add(Out(g + "attn_k_b.weight", [qk_nope, kl, nh], bt, gen_k, None))
    self._add(Out(g + "attn_v_b.weight", [kl, dv, nh], bt, gen_v, None))


Plan.split_kv_b = _split_kv_b


# ===================================================================================== metadata
def add_kv(out, key, value, type_name=None):
    if type_name is None:
        type_name = {bool: "bool", int: "i64", float: "f64", str: "string"}.get(type(value))
    if type_name is None and isinstance(value, (list, tuple)):
        elem = {bool: "bool", int: "i32", float: "f32", str: "string"}.get(type(value[0]))
        type_name = "array:" + elem
    if type_name is None:
        raise TypeError(key)
    out.append((key, type_name, value))


def metadata(cfg, n_export, opts, meta_src):
    m = []
    add_kv(m, "general.architecture", "ernie5_5")
    add_kv(m, "general.name", cfg.get("_name", "ERNIE 5.5 Flash"))
    add_kv(m, "general.quantization_version", 2, "u32")
    add_kv(m, "general.file_type", 7 if opts.experts != "mxfp4" else 38, "u32")

    ratios = [max(r, 0) for r in cfg["compress_ratios"]][:n_export]
    rp = cfg["rope_scaling"]
    kv = {
        "block_count": n_export,
        "context_length": cfg["max_position_embeddings"],
        "embedding_length": cfg["hidden_size"],
        "feed_forward_length": cfg["intermediate_size"],
        "attention.head_count": cfg["num_attention_heads"],
        "attention.head_count_kv": cfg["num_key_value_heads"],
        "attention.key_length": cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"],
        "attention.value_length": cfg["kv_lora_rank"],
        "attention.key_length_swa": cfg["head_dim"],
        "attention.value_length_swa": cfg["head_dim"],
        "attention.key_length_mla": cfg["hybrid_mla_qk_nope_head_dim"] + cfg["hybrid_mla_qk_rope_head_dim"],
        "attention.value_length_mla": cfg["hybrid_mla_v_head_dim"],
        "attention.kv_lora_rank": cfg["kv_lora_rank"],
        "attention.q_lora_rank": cfg["q_lora_rank"],
        "attention.sliding_window": cfg["sliding_window"],
        "attention.layer_norm_rms_epsilon": cfg["rms_norm_eps"],
        "attention.compress_rope_freq_base": cfg["compress_rope_theta"],
        "attention.output_group_count": cfg["o_groups"],
        "attention.output_lora_rank": cfg["o_lora_rank"],
        "attention.indexer.head_count": cfg["index_n_heads"],
        "attention.indexer.key_length": cfg["index_head_dim"],
        "attention.indexer.top_k": cfg["index_topk"],
        "rope.dimension_count": cfg["qk_rope_head_dim"],
        "rope.freq_base": float(cfg["rope_theta"]),   # llama.cpp reads this key as F32
        "rope.scaling.factor": float(rp["factor"]),
        "rope.scaling.original_context_length": int(rp["original_max_position_embeddings"]),
        "rope.scaling.yarn_beta_fast": float(rp["beta_fast"]),
        "rope.scaling.yarn_beta_slow": float(rp["beta_slow"]),
        "hyper_connection.count": cfg["hc_mult"],
        "hyper_connection.sinkhorn_iterations": cfg["hc_sinkhorn_iters"],
        "hyper_connection.epsilon": cfg["hc_eps"],
        "leading_dense_block_count": min(cfg["first_k_dense_replace"], n_export),
        "expert_count": cfg["n_routed_experts"],
        "expert_used_count": cfg["num_experts_per_tok"],
        "expert_feed_forward_length": cfg["moe_intermediate_size"],
        "expert_shared_count": cfg["n_shared_experts"],
        "expert_weights_norm": bool(cfg["norm_topk_prob"]),
        "expert_latent_length": cfg["moe_latent_size"],
        "expert_gating_func": 2,
        "expert_group_count": cfg["n_group"],
        "expert_group_used_count": cfg["topk_group"],
        "activation.situ_beta": float(cfg["activation_situ_beta"]),
        "activation.situ_linear_beta": float(cfg["activation_situ_linear_beta"]),
        "nextn_predict_layers": 0,
    }
    for k, v in kv.items():
        t = {bool: "bool", int: "u32", float: "f32"}.get(type(v))
        add_kv(m, "ernie5_5." + k, v, t)
    add_kv(m, "ernie5_5.attention.compress_ratios", ratios, "array:i32")
    add_kv(m, "ernie5_5.rope.scaling.type", rp["type"])

    multimax = cfg.get("_multimax")
    if multimax is None and meta_src is not None:
        multimax = meta_src["multimax"]
        add_kv(m, "ernie5_5.dev_multimax_from", os.path.basename(meta_src["path"]))
    if multimax is None:
        raise SystemExit("logit_multimax is not in config.json and no --metadata-from GGUF was given "
                         "(Preview's own values live in the last shard)")
    add_kv(m, "ernie5_5.logit_multimax", [float(x) for x in multimax], "array:f32")

    if meta_src is not None:
        tok = meta_src["tokenizer"]
    else:
        tj = os.path.join(opts.src, "tokenizer.json")
        if not os.path.exists(tj):
            raise SystemExit(f"{tj} is not there yet; pass --metadata-from <a GGUF of the same family> "
                             "(the 60G Base shard 1 carries the 201,216-token vocabulary)")
        tok = read_tokenizer_json(tj)
        # The PR's set_vocab: the post-processor prepends <|ERNIE_MASK|> (also the eos), not config's bos,
        # and the pair separator does not apply to single sequences.
        toks = tok["tokenizer.ggml.tokens"][0]
        mask = toks.index("<|ERNIE_MASK|>") if "<|ERNIE_MASK|>" in toks else cfg["bos_token_id"]
        tok.update({
            "tokenizer.ggml.bos_token_id": (mask, "u32"),
            "tokenizer.ggml.eos_token_id": (int(cfg["eos_token_id"] if isinstance(cfg["eos_token_id"], int)
                                               else cfg["eos_token_id"][0]), "u32"),
            "tokenizer.ggml.padding_token_id": (int(cfg.get("pad_token_id", 0)), "u32"),
            "tokenizer.ggml.seperator_token_id": (mask, "u32"),
            "tokenizer.ggml.add_bos_token": (True, "bool"),
            "tokenizer.ggml.add_sep_token": (False, "bool"),
        })
    for k, v in tok.items():
        add_kv(m, k, v[0], v[1])
    if opts.layers:
        add_kv(m, "ernie5_5.dev_truncated_layers", opts.layers, "u32")
    if opts.experts == "skip":
        add_kv(m, "ernie5_5.dev_experts_skipped", True, "bool")
    return m


# The tokenizer keys the converter writes, with their GGUF metadata type.  llama.cpp's own conversion produces
# exactly this block for ERNIE 5.5 (see the 60G Base shard); copying it keeps the writer from re-deriving the
# gpt2 pre-tokenizer rules, which live in llama.cpp, not in this repo.
_TOKEN_KEYS = {
    "tokenizer.ggml.model": "string",
    "tokenizer.ggml.pre": "string",
    "tokenizer.ggml.tokens": "array:string",
    "tokenizer.ggml.token_type": "array:i32",
    "tokenizer.ggml.merges": "array:string",
    "tokenizer.ggml.bos_token_id": "u32",
    "tokenizer.ggml.eos_token_id": "u32",
    "tokenizer.ggml.padding_token_id": "u32",
    "tokenizer.ggml.seperator_token_id": "u32",
    "tokenizer.ggml.add_bos_token": "bool",
    "tokenizer.ggml.add_sep_token": "bool",
    "tokenizer.ggml.clean_spaces": "bool",
}


def read_meta_gguf(path):
    """Copy the tokenizer block and MultiMax values out of another ernie5_5 GGUF."""
    f = GH.GGUFFile(path)
    g = f.metadata
    mm = g.get("ernie5_5.logit_multimax")
    if mm is None:
        raise SystemExit(f"{path}: no ernie5_5.logit_multimax")
    tok = {}
    for k, t in _TOKEN_KEYS.items():
        if k not in g:
            continue
        v = g[k]
        if t == "bool":
            tok[k] = (bool(v), t)
        elif t == "string":
            tok[k] = (str(v), t)
        elif t.startswith("array:"):
            elem = t.split(":", 1)[1]
            tok[k] = (list(v) if elem == "string" else [int(x) for x in v], t)
        else:
            tok[k] = (int(v), t)
    for need in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.pre"):
        if need not in tok:
            raise SystemExit(f"{path}: missing {need}")
    if len(tok["tokenizer.ggml.tokens"][0]) != tok["tokenizer.ggml.token_type"][0].__len__():
        raise SystemExit(f"{path}: tokenizer.ggml.tokens and token_type have different lengths")
    return {"path": path, "multimax": [float(x) for x in mm], "tokenizer": tok}


def read_tokenizer_json(path):
    """Minimal tokenizer.json -> the gpt2 arrays llama.cpp expects (the PR converter's `set_vocab`, without
    transformers).  token_type: 1 normal, 3 control (special added token), 4 user-defined, 6 byte (`<0xHH>`)."""
    with open(path) as f:
        j = json.load(f)
    model = j.get("model", {})
    vocab = model.get("vocab", {})
    if not vocab:
        raise SystemExit(f"{path}: no model.vocab")
    added = {int(t["id"]): t for t in j.get("added_tokens", []) if "id" in t}
    size = max(list(vocab.values()) + list(added.keys())) + 1
    tokens = [None] * size
    types = [1] * size
    for tok, i in vocab.items():
        if 0 <= i < size:
            tokens[i] = tok
    for i, t in added.items():
        if 0 <= i < size:
            tokens[i] = t.get("content", tokens[i])
            types[i] = 3 if t.get("special") else 4
    for i, tok in enumerate(tokens):
        if tok is None:
            raise SystemExit(f"{path}: token id {i} is missing from model.vocab")
        if types[i] == 1 and len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
            types[i] = 6
    merges = model.get("merges", [])
    merges = [m if isinstance(m, str) else " ".join(m) for m in merges]
    if not merges:
        raise SystemExit(f"{path}: no model.merges (gpt2 BPE required)")
    return {
        "tokenizer.ggml.model": ("gpt2", "string"),
        "tokenizer.ggml.pre": ("ernie5", "string"),
        "tokenizer.ggml.tokens": (tokens, "array:string"),
        "tokenizer.ggml.token_type": (types, "array:i32"),
        "tokenizer.ggml.merges": (merges, "array:string"),
    }


# ===================================================================================== writer
def chunk_bytes(meta, tensors):
    """Serialise the header: metadata records, tensor directory, alignment pad."""
    out = bytearray()
    out += struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(meta))
    for key, t, v in meta:
        out += struct.pack("<Q", len(key)) + key.encode() + _kv(key, t, v)
    off = 0
    for t in tensors:
        out += struct.pack("<Q", len(t.name)) + t.name.encode()
        out += struct.pack("<I", len(t.shape)) + struct.pack(f"<{len(t.shape)}Q", *t.shape)
        out += struct.pack("<I", TYPE_ID[t.gtype]) + struct.pack("<Q", off)
        off += align(t.nbytes)
    out += bytes(align(len(out)) - len(out))
    return bytes(out)


def _kv(key, t, v):
    out = bytearray()
    if t.startswith("array:"):
        elem = t.split(":", 1)[1]
        out += struct.pack("<I", META_ID["array"]) + struct.pack("<I", META_ID[elem]) + struct.pack("<Q", len(v))
        for x in v:
            out += _scalar(elem, x)
        return bytes(out)
    out += struct.pack("<I", META_ID[t]) + _scalar(t, v)
    return bytes(out)


def _scalar(t, v):
    if t == "string":
        b = v.encode()
        return struct.pack("<Q", len(b)) + b
    if t == "bool":
        return struct.pack("<B", 1 if v else 0)
    fmt = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i",
           "f32": "<f", "u64": "<Q", "i64": "<q", "f64": "<d"}[t]
    return struct.pack(fmt, v)


class DirectWriter:
    """Sequential file writer for big outputs: O_DIRECT pwrites of a 4096-aligned staging buffer (default 256 MiB), so a
    rotational or exFAT target sees a few large writes per GB instead of page-cache writeback interleaved with the
    source reads.  `sync()` makes everything written so far durable (the unaligned tail goes out as one padded page
    that the next flush rewrites) and returns the durable logical offset.  Falls back to buffered I/O when the
    filesystem refuses O_DIRECT or ERNIE_GGUF_DIRECT=0.  `start` > 0 resumes: the file is cut to `start` and the partial
    page before it is read back into the buffer."""
    ALIGN = 4096

    def __init__(self, path, start=0, buf_mib=256):
        self.path = path
        self.cap = buf_mib << 20
        self.buf = mmap.mmap(-1, self.cap)          # page-aligned anonymous memory
        self.mv = memoryview(self.buf)
        self.direct = os.environ.get("ERNIE_GGUF_DIRECT", "1") != "0" and hasattr(os, "O_DIRECT")
        flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if start == 0 else 0)
        base = start - start % self.ALIGN
        head = b""
        if start:
            with open(path, "rb") as f:            # the partial page before `start`
                f.seek(base)
                head = f.read(start - base)
            if len(head) != start - base:
                raise IOError(f"{path}: cannot read back {start - base} B at {base}")
        self.fd = None
        if self.direct:
            try:
                self.fd = os.open(path, flags | os.O_DIRECT, 0o664)
            except OSError as e:
                log(f"O_DIRECT refused on {path} ({e}); buffered writes")
                self.direct = False
        if self.fd is None:
            self.fd = os.open(path, flags, 0o664)
        if start:
            os.ftruncate(self.fd, start)
        self.flushed = base                          # file offset of buf[0] (always aligned)
        self.n = len(head)                           # bytes held in buf
        self.buf[:self.n] = head
        log(f"writer: {'O_DIRECT' if self.direct else 'buffered'}, {buf_mib} MiB buffer, start {start}")

    def tell(self):
        return self.flushed + self.n

    def _pwrite(self, length):
        done = 0
        while done < length:
            k = os.pwrite(self.fd, self.mv[done:length], self.flushed + done)
            if k <= 0:
                raise IOError(f"{self.path}: pwrite returned {k}")
            done += k

    def _flush_full(self):
        """Write the aligned prefix of the buffer and keep the unaligned remainder."""
        k = self.n - self.n % self.ALIGN
        if not k:
            return
        self._pwrite(k)
        rem = self.n - k
        if rem:
            self.buf[:rem] = self.buf[k:self.n]
        self.flushed += k
        self.n = rem

    def write(self, b):
        mv = memoryview(b).cast("B")
        i = 0
        while i < len(mv):
            k = min(self.cap - self.n, len(mv) - i)
            self.buf[self.n:self.n + k] = mv[i:i + k]
            self.n += k
            i += k
            if self.n == self.cap:
                self._flush_full()

    def sync(self):
        self._flush_full()
        if self.n:   # one padded page for the tail; it stays in the buffer and is rewritten by the next flush
            pad = self.ALIGN - self.n
            self.buf[self.n:self.n + pad] = bytes(pad)
            self._pwrite(self.ALIGN)
        os.fsync(self.fd)
        return self.tell()

    def close(self, final_size=None):
        end = self.sync()
        os.ftruncate(self.fd, end if final_size is None else final_size)
        os.fsync(self.fd)
        os.close(self.fd)
        self.mv.release()
        self.buf.close()
        return end


def write_gguf(path, meta, tensors, opts):
    """Write `path` via `path.partial`, resumable.  After every tensor >= 64 MiB (and the last) the data is
    fsync'd and `path.partial.ckpt` records {header sha256, tensors done, byte offset}.  A rerun with the same
    plan resumes from the checkpoint (header bytes must match); `--resume-at N` resumes a pre-checkpoint
    `.partial` at tensor N after checking its header; `--no-resume` starts over."""
    import hashlib
    import json
    hdr = chunk_bytes(meta, tensors)
    hsha = hashlib.sha256(hdr).hexdigest()
    total = len(hdr) + sum(align(t.nbytes) for t in tensors)
    log(f"{len(tensors)} tensors, {total / 1e9:.2f} GB -> {path}")
    tmp = path + ".partial"
    ckpt = tmp + ".ckpt"
    if os.path.exists(path) and not opts.force:
        raise SystemExit(f"{path} exists (pass --force to replace it)")

    def off_of(k):   # byte offset where tensor k starts
        return len(hdr) + sum(align(t.nbytes) for t in tensors[:k])

    start = 0
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    resume_at = getattr(opts, "resume_at", None)
    if have and not getattr(opts, "no_resume", False):
        if resume_at is None and os.path.exists(ckpt):
            c = json.load(open(ckpt))
            if c.get("header_sha256") != hsha:
                raise SystemExit(f"{ckpt}: header differs from this plan (other args?); pass --no-resume to restart")
            start = int(c["done"])
        elif resume_at is not None:
            start = max(0, min(int(resume_at), len(tensors)))
        if start:
            with open(tmp, "rb") as fp:
                if fp.read(len(hdr)) != hdr:
                    raise SystemExit(f"{tmp}: header bytes differ from this plan; pass --no-resume to restart")
            if have < off_of(start):
                raise SystemExit(f"{tmp}: {have} B < offset {off_of(start)} of tensor {start}; lower --resume-at")
    free = _free(path)
    need = total - (off_of(start) if start else 0)
    if free is not None and free < need + (4 << 30):
        raise SystemExit(f"{path}: needs {need / 2**30:.1f} GiB + 4 GiB reserve, {free / 2**30:.1f} GiB free")

    def save_ckpt(fp, done):
        off = fp.sync()
        with open(ckpt + ".tmp", "w") as cf:
            json.dump({"header_sha256": hsha, "done": done, "offset": off, "n": len(tensors)}, cf)
            cf.flush()
            os.fsync(cf.fileno())
        os.replace(ckpt + ".tmp", ckpt)

    t0 = time.time()
    if start:
        log(f"RESUME at tensor {start}/{len(tensors)} ({tensors[start - 1].name} done), offset {off_of(start)}")
        fp = DirectWriter(tmp, off_of(start))
    else:
        fp = DirectWriter(tmp, 0)
        fp.write(hdr)
        save_ckpt(fp, 0)
    try:
        for i in range(start, len(tensors)):
            t = tensors[i]
            n = 0
            for b in t.emit():
                fp.write(b)
                n += len(b)
            if n != t.nbytes:
                raise ValueError(f"{t.name}: wrote {n} bytes, planned {t.nbytes}")
            fp.write(bytes(align(t.nbytes) - t.nbytes))
            if t.nbytes >= (64 << 20) or i + 1 == len(tensors):
                save_ckpt(fp, i + 1)
                log(f"  [{i + 1}/{len(tensors)}] {t.name} {t.gtype} {t.nbytes / 2**20:.0f} MiB "
                    f"{time.time() - t0:.1f}s")
        if fp.tell() != total:
            raise ValueError(f"{tmp}: ended at {fp.tell()} B, planned {total}")
    except BaseException:
        fp.sync()   # keep what was written durable for the resume; the .ckpt marks the last whole tensor
        raise
    fp.close(total)
    os.replace(tmp, path)
    try:
        os.remove(ckpt)
    except OSError:
        pass
    log(f"done in {time.time() - t0:.1f}s; peak RSS {_rss() / 2**20:.0f} MiB")


def _free(path):
    try:
        import shutil
        return shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
    except OSError:
        return None


def _rss():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


# ===================================================================================== DSpark sidecar
def plan_dspark(cfg, opts, meta_src):
    """The Preview DSpark drafts as a sidecar.  Names follow ernie_dspark_expected_tensors()."""
    store = Store(opts.src, max_open=4)
    p = opts.src + "/config.json"
    ct = json.load(open(p))
    nd = 3
    dh = ct["head_dim"]
    D = ct["hidden_size"]
    nh = ct["num_attention_heads"]
    rq = ct["q_lora_rank"]
    og, ol = ct["o_groups"], ct["o_lora_rank"]
    rank = 256
    V = ct["vocab_size"]
    fe = ct["moe_intermediate_size"]
    lat = ct["moe_latent_size"]
    ne = ct["n_routed_experts"]
    mhc = ct["hc_mult"]
    hcd, hcm = mhc * D, (2 + mhc) * mhc
    exp_type = {"q8_0": "Q8_0", "mxfp4": "MXFP4", "skip": None}[opts.experts]

    pl = Plan(store, opts)
    pre = "model.mtp."
    mtp = [n for n in store.index if n.startswith(pre)]
    if not mtp:
        raise SystemExit("no model.mtp.* tensors: the DSpark drafts live in the last FP8 shard, which is not "
                         "downloaded yet")

    def src_name(cands):
        for c in cands:
            if c in store.index:
                return c
        return None

    main_proj = src_name([pre + "main_proj.weight", pre + "feature_projection.weight"])
    main_norm = src_name([pre + "main_norm.weight", pre + "feature_norm.weight"])
    if main_proj:
        pl.mat(main_proj, "dspark.main_proj.weight", [3 * D, D], "Q8_0")
    if main_norm:
        pl.small([main_norm], "dspark.main_norm.weight", [D], "F32")
    if pre + "norm.weight" in store.index:
        pl.small([pre + "norm.weight"], "dspark.norm.weight", [D], "F32")
    for leaf, dst, shape, fn in (
            ("hc_head_fn", "dspark.hc_head_fn.weight", [hcd, mhc], lambda a: a[0].T.copy()),
            ("hc_head_base", "dspark.hc_head_base.weight", [mhc], lambda a: a[0].reshape(-1)),
            ("hc_head_scale", "dspark.hc_head_scale.weight", [1], lambda a: a[0].reshape(-1))):
        src = src_name([pre + leaf, pre + "mhc_contract." + leaf])   # the Preview nests it under mhc_contract
        if src:
            pl.small([src], dst, shape, "F32", fn)
    markov_w1 = src_name([pre + "markov_head.markov_w1.weight", pre + "markov_head.embed.weight"])
    markov_w2 = src_name([pre + "markov_head.markov_w2.weight", pre + "markov_head.head.weight"])
    if markov_w1:
        pl.small([markov_w1], "dspark.markov_head.embed.weight", [rank, V], "Q8_0", lambda a: a[0].T.copy())
    if markov_w2:
        pl.small([markov_w2], "dspark.markov_head.head.weight", [rank, V], "Q8_0")   # stored [V, 256] = the Linear layout already (26 Nov: the .T scrambled it)
    conf = src_name([pre + "confidence_head.projection.weight", pre + "confidence_head.proj.weight"])
    if conf:
        pl.small([conf], "dspark.confidence_head.proj.weight", [D + rank, 1], "F32")

    # shared context K/V = each draft block's linear_kv_proj, one column shard per block
    kv_names = [f"{pre}{i}.self_attn.linear_kv_proj.weight" for i in range(nd)]
    if all(n in store.index for n in kv_names):
        def gen_ctx():
            for n in kv_names:
                rows = store.index[n].shape[0]          # head_dim rows of D
                for r0 in range(0, rows, 128):
                    yield encode(store.f32(n, r0, min(rows, r0 + 128)), "Q8_0")
        pl._add(Out("dspark.context_wkv_proj.weight", [D, nd * dh], "Q8_0", gen_ctx, None))

    for i in range(nd):
        p = f"{pre}{i}."
        g = f"dspark.blk.{i}."
        pl.maybe_small_or_zero(p + "input_layernorm.weight", g + "attn_norm.weight", [D], "F32")
        pl.maybe_small_or_zero(p + "post_attention_layernorm.weight", g + "ffn_norm.weight", [D], "F32")
        for kind, sk in (("attn", "self_attention"), ("ffn", "mlp")):
            base = p + f"{sk}_hyper_connection."
            if base + "mapping_proj.weight" in store.index:
                pl.small([base + "mapping_proj.weight"], g + f"hc_{kind}_fn.weight", [hcd, hcm], "F32",
                         lambda a: a[0].T.copy())
                pl.small([base + "bias"], g + f"hc_{kind}_base.weight", [hcm], "F32")
                pl.small([base + "alpha_pre", base + "alpha_post", base + "alpha_res"], g + f"hc_{kind}_scale.weight",
                         [3], "F32", lambda a: np.concatenate([x.reshape(1) for x in a]))
        if p + "self_attn.core_attention.attn_sink" in store.index:
            pl.small([p + "self_attn.core_attention.attn_sink"], g + "attn_sinks.weight", [nh], "F32")
        if p + "self_attn.vha_postmix_U" in store.index:
            pl.small([p + "self_attn.vha_postmix_U", p + "self_attn.vha_postmix_V"], g + "attn_vha.weight",
                     [nh, nh], "F32", fold_vha)
        if p + "self_attn.linear_q_down_proj.weight" in store.index:
            pl.mat(p + "self_attn.linear_q_down_proj.weight", g + "attn_q_a.weight", [D, rq], "Q8_0")
            pl.mat(p + "self_attn.linear_q_up_proj.weight", g + "attn_q_b.weight", [rq, nh * dh], "Q8_0")
            pl.small([p + "self_attn.kv_layernorm.weight"], g + "attn_kv_a_norm.weight", [dh], "F32")
            pl.mat(p + "self_attn.linear_o_group_proj", g + "attn_output_a.weight", [nh * dh // og, ol, og], "Q8_0")
            pl.mat(p + "self_attn.o_proj.weight", g + "attn_output_b.weight", [og * ol, D], "Q8_0")
            pl.mat(p + "self_attn.gate_proj.weight", g + "attn_gate.weight", [rq, og * ol], "Q8_0")
            pl.small([p + "self_attn.q_layernorm.weight"], g + "attn_q_a_norm.weight", [rq], "F32")
        # latent MoE
        if p + "block_sparse_moe.gate.weight" in store.index:
            pl.small([p + "block_sparse_moe.gate.weight", p + "block_sparse_moe.gate.weight_1"],
                     g + "ffn_gate_inp.weight", [D, 2 * ne], "F32", lambda a: np.concatenate(a, axis=0))
            pl.mat(p + "block_sparse_moe.fc1_latent_proj.weight", g + "ffn_routed_down.weight", [D, lat], "Q8_0")
            pl.mat(p + "block_sparse_moe.fc2_latent_proj.weight", g + "ffn_routed_up.weight", [lat, D], "Q8_0")
            pl.small([p + "block_sparse_moe.latent_norm.weight"], g + "ffn_routed_norm.weight", [lat], "F32")
            esc = p + "block_sparse_moe.e_score_correction_bias"
            if esc not in store.index:
                esc = p + "block_sparse_moe.e_score_correction.bias"
            pl.small([esc], g + "exp_probs_b.bias", [ne], "F32")
            pl.small([p + "block_sparse_moe.gate.routed_scaling_factor_param"], g + "exp_scale.weight", [ne], "F32")
            pl.mat(p + "block_sparse_moe.shared_experts.w1.weight", g + "ffn_gate_shexp.weight", [D, fe], "Q8_0")
            pl.mat(p + "block_sparse_moe.shared_experts.w2.weight", g + "ffn_down_shexp.weight", [fe, D], "Q8_0")
            pl.mat(p + "block_sparse_moe.shared_experts.w3.weight", g + "ffn_up_shexp.weight", [D, fe], "Q8_0")
        if exp_type:
            for src_role, dst_role, shape in (("w1", "ffn_gate_exps", [lat, fe, ne]),
                                              ("w2", "ffn_down_exps", [fe, lat, ne]),
                                              ("w3", "ffn_up_exps", [lat, fe, ne])):
                names = [f"{p}block_sparse_moe.experts.{e}.{src_role}.weight" for e in range(ne)]
                if all(n in store.index for n in names):
                    pl.stacked(names, g + dst_role + ".weight", shape, exp_type, fe, lat)

    unknown = [n for n in mtp if not _dspark_known(n, nd)]
    if unknown:
        raise SystemExit("unrecognised DSpark tensors (the sidecar map would drop them):\n  " + "\n  ".join(unknown[:20]))
    meta = [("general.architecture", "string", "ernie5_5-dspark"),
            ("general.name", "string", ct.get("_name", "ERNIE 5.5 Flash") + " DSpark drafts"),
            ("ernie5_5.dspark.n_draft", "u32", nd),
            ("ernie5_5.dspark.block_size", "u32", cfg.get("dspark_block_size", 7)),
            ("ernie5_5.dspark.markov_rank", "u32", cfg.get("dspark_markov_rank", 256)),
            ("ernie5_5.dspark.noise_token", "u32", cfg.get("dspark_noise_token_id", 1024)),
            ("ernie5_5.dspark.target_layers", "array:i32", cfg.get("dspark_target_layer_ids", [40, 41, 42]))]
    if os.path.exists(opts.dspark_out) and not opts.force:
        log(f"{opts.dspark_out} already complete; skipping")
        return
    write_gguf(opts.dspark_out, meta, pl.out, opts)


def _dspark_known(name, nd):
    pre = "model.mtp."
    rest = name[len(pre):]
    if rest.startswith(tuple(f"{i}." for i in range(nd))):
        rest = rest.split(".", 1)[1]
    elif rest.startswith("markov_head.") or rest.startswith("confidence_head."):
        return True
    else:
        return rest.split(".", 1)[0] in ("main_proj", "main_norm", "norm", "hc_head_fn", "hc_head_base",
                                         "hc_head_scale", "feature_projection", "feature_norm", "mhc_contract")
    for hc in ("self_attention_hyper_connection.", "mlp_hyper_connection."):
        if rest.startswith(hc):   # plan_dspark writes these as hc_{attn,ffn}_{fn,base,scale}
            return rest[len(hc):] in ("mapping_proj.weight", "bias", "alpha_pre", "alpha_post", "alpha_res")
    return rest.split(".", 1)[0] in {
        "input_layernorm", "post_attention_layernorm", "self_attn", "block_sparse_moe"}


# ===================================================================================== verify
_Q8 = None
_MX = None


def dequant(gtype, raw, shape):
    global _Q8, _MX
    a = np.frombuffer(raw, np.uint8)
    total = int(np.prod(shape))
    if gtype == "F32":
        return np.frombuffer(raw, "<f4").astype(np.float32)
    if gtype == "F16":
        return np.frombuffer(raw, "<f2").astype(np.float32)
    if gtype == "Q8_0":
        nb = total // 32
        b = a.reshape(nb, 34)
        d = b[:, :2].copy().view("<f2").astype(np.float32)
        return (b[:, 2:].copy().view(np.int8).astype(np.float32) * d).reshape(-1)
    if gtype == "MXFP4":
        nb = total // 32
        b = a.reshape(nb, 17)
        e = b[:, 0:1]
        bits = np.where(e < 2, np.uint32(0x00200000) << e.astype(np.uint32),
                        (e.astype(np.uint32) - 1) << np.uint32(23))
        d = bits.view(np.float32)
        qs = b[:, 1:].astype(np.uint8)
        lo = qs & 0x0F
        hi = qs >> 4
        vals = np.empty((nb, 32), np.float32)
        vals[:, :16] = KVALUES.astype(np.float32)[lo]
        vals[:, 16:] = KVALUES.astype(np.float32)[hi]
        return (vals * d).reshape(-1)
    raise ValueError(gtype)


def verify(opts):
    f = GH.GGUFFile(opts.verify)
    tensors = {t.name: t for t in f.tensors}
    store = Store(opts.src)
    cfg = read_config(opts.src)
    pl = Plan(store, argparse.Namespace(**{**vars(opts), "check_source": False}))
    plan_model(pl, cfg, opts._n_export)
    data = open(opts.verify, "rb")

    def read_payload(t, nbytes=None):
        nb = t.expected_bytes()
        if nb is None:
            raise ValueError(t.name)
        nb = nb if nbytes is None else min(nb, nbytes)
        data.seek(f.data_start + t.offset)
        return data.read(nb)

    samples = []
    for o in pl.out:
        if o.sample and o.gtype in ("Q8_0", "MXFP4") and "attn_q_b" in o.name:
            samples.append(("backbone FP8", o))
            break
    for o in pl.out:
        if o.sample and "ffn_gate_exps" in o.name:
            samples.append(("expert FP8", o))
            break
    for o in pl.out:
        if o.sample and o.gtype == "F32" and o.name.endswith("attn_q_a_norm.weight"):
            samples.append(("BF16 norm", o))
            break
    if len(samples) < 3:
        raise SystemExit("could not find the three sample tensors in the plan")
    worst = 0.0
    for label, o in samples:
        t = tensors.get(o.name)
        if t is None:
            raise SystemExit(f"{o.name} is not in the GGUF")
        src, per_rows = o.sample
        if per_rows is None:
            got = dequant(o.gtype, read_payload(t), t.shape)
            ref = store.f32(src).reshape(-1)
        else:
            # one expert: its bytes are the first per_rows * shape[0] values of the stack
            got = dequant(o.gtype, read_payload(t, nbytes_for([t.shape[0], per_rows], o.gtype)),
                          [t.shape[0], per_rows])
            ref = store.f32(src, 0, per_rows).reshape(-1)
        rel = float(np.linalg.norm(got - ref) / max(1e-12, np.linalg.norm(ref)))
        worst = max(worst, rel)
        log(f"  {label:12s} {o.name:34s} {o.gtype:6s} rel L2 {rel:.3e}")
    log(f"  worst rel L2 {worst:.3e} ({'PASS' if worst < 1e-2 else 'FAIL'} at 1e-2)")
    return 0 if worst < 1e-2 else 1


# ===================================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description="ERNIE 5.5 Flash FP8 safetensors -> ernie5_5 GGUF (streaming, CPU)")
    ap.add_argument("src", nargs="?", help="the FP8 repo directory (config.json + model-*.safetensors)")
    ap.add_argument("-o", "--out", help="output GGUF path (refuses to overwrite without --force)")
    ap.add_argument("--layers", type=int, default=0, help="export only blocks 0..N-1 (0 = all)")
    ap.add_argument("--experts", choices=("q8_0", "mxfp4", "f16", "f32", "skip"), default="q8_0",
                    help="routed-expert type (skip = trunk-only file; f16/f32 = unquantised oracle files)")
    ap.add_argument("--ftype", choices=("q8_0", "f16", "f32"), default="q8_0",
                    help="backbone/dense matrix type (f16/f32 = unquantised oracle files that reproduce the "
                         "FP8 values exactly)")
    ap.add_argument("--dspark-out", help="write the DSpark drafts to this separate sidecar GGUF")
    ap.add_argument("--metadata-from", help="GGUF of the same family to copy logit_multimax + tokenizer from "
                                           "(Preview's own are in the last shard, not downloaded yet)")
    ap.add_argument("--check-source", action="store_true", help="validate every complete shard header, write nothing")
    ap.add_argument("--verify", help="GGUF produced by this tool: numeric spot check against the FP8 source")
    ap.add_argument("--force", action="store_true", help="replace an existing output file")
    ap.add_argument("--no-dspark", action="store_true", help="ignore --dspark-out")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing .partial/.ckpt and start over")
    ap.add_argument("--resume-at", type=int, help="resume a .partial that has no .ckpt at tensor N (0-based count "
                    "of tensors already complete; take the last logged [n/total] minus a margin)")
    opts = ap.parse_args(argv)
    if not opts.src:
        ap.error("SRC is required")
    if not os.path.exists(os.path.join(opts.src, "config.json")):
        ap.error(f"{opts.src}: no config.json")
    cfg = read_config(opts.src)
    cfg["_name"] = os.environ.get("ERNIE_SRC_NAME", "ERNIE 5.5 Flash Preview")
    opts._n_export = opts.layers if opts.layers else cfg["num_hidden_layers"]
    if opts.layers > cfg["num_hidden_layers"]:
        ap.error(f"--layers {opts.layers} > num_hidden_layers {cfg['num_hidden_layers']}")

    store = Store(opts.src)
    log(f"{len(store.files)} complete shards / {len(store.index)} tensors"
        + (f" ({len(store.skipped)} incomplete shards skipped)" if store.skipped else ""))

    if opts.verify:
        return verify(opts)

    pl = Plan(store, opts)
    plan_model(pl, cfg, opts._n_export)
    if opts.check_source:
        import re as _re

        def in_scope(n):
            m = _re.match(r"model\.layers\.(\d+)\.", n)
            return m is None or int(m.group(1)) < opts._n_export

        # DSpark drafts are a separate sidecar (`--dspark-out`), never part of the trunk file.
        dspark = sorted(n for n in store.index if n.startswith("model.mtp."))
        unknown = sorted(n for n in store.index
                         if in_scope(n) and n not in pl.saw and not n.startswith("model.mtp."))
        log(f"check-source: {len(pl.saw)} source tensors consumed, {len(pl.missing)} missing, "
            f"{len(unknown)} present but unrecognised, {len(dspark)} dspark (not exported)")
        for n in pl.missing[:20]:
            log(f"  missing: {n}")
        for n in unknown[:20]:
            log(f"  unknown: {n}")
        # A shard that has not downloaded is not a mapping error; an unknown name or a shape the map cannot
        # consume is.
        ok = not unknown
        log("source header check " + ("PASS" if ok else "FAIL") +
            ("" if not pl.missing else f" ({len(pl.missing)} tensors not downloaded yet)"))
        return 0 if ok else 1

    # A truncated export may only fill ROOT tensors with zeros (their shard is the last one).  A missing block
    # tensor means the layer's shard is not complete and the file would be structurally broken.
    root = ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight")
    block_missing = [n for n in pl.missing
                     if not (n.startswith(root) or n.startswith("model.mhc_contract."))]
    if block_missing:
        raise SystemExit("incomplete shards: these block tensors are not on disk:\n  " +
                         "\n  ".join(block_missing[:20]))
    if not opts.out:
        ap.error("-o/--out is required")
    meta_src = read_meta_gguf(opts.metadata_from) if opts.metadata_from else None
    meta = metadata(cfg, opts._n_export, opts, meta_src)
    if pl.placeholders:
        log(f"note: {len(pl.placeholders)} root tensors are placeholders (their shard is not downloaded): "
            + ", ".join(pl.placeholders[:6]))
        meta.append(("ernie5_5.dev_placeholder_tensors", "array:string", pl.placeholders))
    if os.path.exists(opts.out) and not opts.force:
        log(f"{opts.out} already complete; skipping the trunk (resume run)")
    else:
        write_gguf(opts.out, meta, pl.out, opts)
        store.report_io()

    if opts.dspark_out and not opts.no_dspark:
        plan_dspark(cfg, opts, meta_src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
