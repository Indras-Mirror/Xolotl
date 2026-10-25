"""ds41_vision.py - a resident image encoder for DeepSeek-V4.1-Flash, a drop-in for Strata's
`strata-vision` process.

The process it replaces is `tools/vision/strata_vision.cpp` (PolyStrata; the DeepSeek/Strata fork is
identical), driven by `serve/server.py` (`class Vision`, ~line 1262) and consumed by the engine's
GENI image path (`src/program/generate.cpp`, ~line 5752 and 6313).  Same argv shape, same stdin/stdout
line protocol, same output-file format.

PROTOCOL (strata_vision.cpp:12-16, 184-233)
  argv   : --mmproj <sidecar.gguf> --model <text model .gguf> [--gpu] [--threads N] [--max-tokens N]
           [--min-tokens N] [--flash-attn on|off|auto]        (server.py:1268 builds exactly this)
           [--vision-weights <tower.safetensors | checkpoint dir | mmproj.gguf>]   (serve_ds41.py adds it)
  stdout : "READY <n_embd>\\n" once (server.py:1297), then per stdin line
             "ENC <image path> <output path>"  ->  "OK <n> <nx> <ny> <ms>"  or  "ERR <message>"
             "QUIT"                            ->  exit 0
           The server reads `line.split()[1]` as the number of <|image_pad|> tokens to splice in
           (server.py:1384, server.py:2294).

OUTPUT FILE - the engine's SVE1 record (generate.cpp:6319-6331)
  int32 {0x31455653 'SVE1', n, nx, ny, n_embd}  then  float32[n][n_embd], little-endian host order.
  Row i sits at grid (x = i % nx, y = i / nx); the engine requires nx*ny == n.  Several records are
  concatenated for a multi-image request (server.py:2303-2306), so one file per image is right.

THE SPAN - exactly what model.py's `merge_image_embeddings` writes (model.py:1228-1239,
image_processor.py:132-137):
  n = num_image_tokens(n_llm_h, n_llm_w) = n_llm_h*(n_llm_w+1) + 2 rows of `dim` (5120) =
      [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
  IMAGE rows are the Aligner output in reading order; the delimiter rows are the learned
  image_start / image_newline / image_end vectors.  Weights come from the sidecar GGUF
  (vision.* / aligner.* / image_*), the ViT + Aligner classes are imported from DeepSeek's reference
  `vision.py` unchanged.  Preprocessing is DeepSeek's own `image_processor.load_image`.

ADAPTED (documented deviations from strata_vision.cpp)
  * Weights: the C++ reads an mtmd mmproj; here the same tensors live in the V4.1 sidecar GGUF.
  * --vision-weights (serve config: vision "weights"): the sidecar keeps the tower in Q8_0, and Q8_0 of the
    checkpoint's BF16 alone moves some rows to cosine 0.996 against the reference (test_ds41_vision.py, 300x1200).
    The full-precision tower - a safetensors file with the checkpoint's vision.* / aligner.* / image_* bytes,
    the checkpoint directory itself, or an mmproj GGUF (clip / deepseek41v, names mapped back) - replaces the
    sidecar's tensors; the sidecar still gives the vision config (cross-checked).  Same memory and speed: the
    encoder dequantizes to --dtype at load either way.
  * Grid: DeepSeek's span is n_llm_h rows of (n_llm_w+1) plus the 2 delimiters, so nx*ny == n cannot
    be the reading-order rectangle (that would need n = n_llm_h*(n_llm_w+1) = n-2).  We emit the flat
    reading order nx = n, ny = 1. V4.1 uses sequential 1-D text positions for the whole span;
    nx/ny do not change the engine's RoPE positions.
  * No mtmd flash-attn / min-tokens knobs (accepted and ignored).
  * The server already caches by image hash (server.py:1363-1388), so this process keeps no cache.

ON THE GPU (--device cuda / --gpu; measured on an RTX 5090)
  * The server starts the encoder BEFORE the engine and waits for READY (server.py `vision = Vision(...)` ahead of
    "loading the model"); the ds41 engine sizes its VRAM expert cache from the card's free memory at every re-seed
    (ds41_generate.cpp:419, :855 - free minus --vram-margin).  So everything the encoder will ever hold must be on the
    card before READY: the weights, the CUDA context, and the activation pool of the largest image.  The encoder
    runs one warm-up of the largest grid --max-tokens allows (--warmup-tokens) and keeps torch's caching allocator
    pool (no empty_cache): later images reuse it, the engine never sees the encoder grow under its cache.
  * --offload: the weights stay in pinned host RAM and are copied to the card per ENC, then freed with the pool.
    Only the CUDA context (~0.5 GiB) stays resident.  NOTE: the ds41 engine fills the card to --vram-margin at each
    re-seed and has no "keep N MiB free" command (the server's elastic mode needs one), so --offload needs a
    --vram-margin that covers the encoder's peak - which costs the same slots as the resident encoder.
  * --tf32: TF32 tensor cores for the f32 matmuls (off by default: the encoder is exact f32 unless asked).
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch

# The engine's SVE1 magic and the GGML type ids we can dequantize.
SVE1 = 0x31455653
F32, F16, Q8_0, BF16 = 0, 1, 8, 30
BYTES_PER_ELEM = {F32: 4, F16: 2, BF16: 2}

DEFAULT_SIDECAR = "/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf"
DEFAULT_REFERENCE = os.environ.get("DS41_REFERENCE", os.path.expanduser("~/AI/gsq-rco/dsv41-inference"))
VISION_PREFIXES = ("vision.", "aligner.", "image_")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------- GGUF (header only)
_GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_gguf_header(path):
    """-> (kv, [(name, shape, ggml_type, offset)], data_start).  Offsets are relative to data_start."""
    with open(path, "rb") as f:
        rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))

        def rs():
            (n,) = rd("<Q")
            return f.read(n).decode("utf-8", "replace")

        def val(ty):
            if ty == 8:                       # GGUF_STRING
                return rs()
            if ty == 9:                       # GGUF_ARRAY
                et, n = rd("<IQ")
                return [val(et) for _ in range(n)] if et in (8, 9) else list(rd("<" + str(n) + _GGUF_SCALARS[et]))
            return rd("<" + _GGUF_SCALARS[ty])[0]

        assert f.read(4) == b"GGUF", f"{path}: not a GGUF file"
        (ver,) = rd("<I")
        assert ver == 3, f"{path}: GGUF v{ver}, expected v3"
        n_tensors, n_kv = rd("<QQ")
        kv = {}
        for _ in range(n_kv):
            k = rs()
            (ty,) = rd("<I")
            kv[k] = val(ty)
        tensors = []
        for _ in range(n_tensors):
            name = rs()
            (nd,) = rd("<I")
            shape = rd(f"<{nd}Q")
            ty, off = rd("<IQ")
            tensors.append((name, tuple(int(x) for x in shape), int(ty), int(off)))
        align = int(kv.get("general.alignment", 32))
        pos = f.tell()
        data_start = (pos + align - 1) // align * align
    return kv, tensors, data_start


def ggml_nbytes(shape, ty):
    n = int(np.prod(shape)) if shape else 0
    if ty in BYTES_PER_ELEM:
        return n * BYTES_PER_ELEM[ty]
    if ty == Q8_0:
        assert n % 32 == 0, f"Q8_0 row not a multiple of 32 ({shape})"
        return n // 32 * 34
    raise ValueError(f"unsupported GGML type {ty}")


def dequantize(raw, ty):
    """Dequantize one ggml tensor to a flat float32 numpy array in ggml order (ne0 fastest,
    i.e. C-order over the reversed shape)."""
    if ty == F32:
        return np.frombuffer(raw, np.float32).astype(np.float32)
    if ty == F16:
        return np.frombuffer(raw, np.float16).astype(np.float32)
    if ty == BF16:
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    if ty == Q8_0:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 34)
        d = np.ascontiguousarray(b[:, :2]).view(np.float16).astype(np.float32).reshape(-1)
        q = np.ascontiguousarray(b[:, 2:]).view(np.int8).astype(np.float32)
        return (d[:, None] * q).reshape(-1).astype(np.float32)
    raise ValueError(f"unsupported GGML type {ty}")


def _pread(fd, n, off):
    if hasattr(os, "pread"):
        return os.pread(fd, n, off)
    os.lseek(fd, off, os.SEEK_SET)   # Windows has no os.pread; one reader per process, so seek + read is safe
    return os.read(fd, n)


def _pread_all(fd, nbytes, off, name):
    """All `nbytes` at `off` (looping on short reads, through the platform-aware _pread), or OSError naming `name`."""
    raw = bytearray()
    while len(raw) < nbytes:
        b = _pread(fd, nbytes - len(raw), off + len(raw))
        if not b:
            raise OSError(f"{name}: short read")
        raw += b
    return bytes(raw)


def _open_ro(path):
    return os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))   # O_BINARY: Windows opens text mode otherwise


class SidecarWeights:
    """Reads only the vision.* / aligner.* / image_* tensors of the sidecar GGUF, by offset."""

    def __init__(self, path):
        self.kv, tensors, self.data_start = read_gguf_header(path)
        self.path = path
        self.tensors = {n: (s, t, o) for n, s, t, o in tensors if n.startswith(VISION_PREFIXES)}
        if not self.tensors:
            raise ValueError(f"{path}: no vision.* / aligner.* / image_* tensors")
        self.fd = _open_ro(path)

    def names(self):
        return sorted(self.tensors)

    def get(self, name):
        """-> float32 numpy array in the checkpoint's own HF orientation (C-order over ne0, so
        [out, in] for a Linear weight -- the reverse of the ggml header shape)."""
        shape, ty, off = self.tensors[name]
        raw = _pread_all(self.fd, ggml_nbytes(shape, ty), self.data_start + off, name)
        return dequantize(raw, ty).reshape(shape[::-1]).astype(np.float32)

    def torch(self, name, tp, dtype):
        return tp.from_numpy(np.ascontiguousarray(self.get(name))).to(dtype)

    @property
    def vision_config(self):
        return json.loads(self.kv["deepseek41.sidecar.vision_config"])

    @property
    def image_token_id(self):
        return self.kv.get("deepseek41.sidecar.image_token_id")

    def describe(self):
        types = sorted({_GGML_NAMES.get(t, str(t)) for _, t, _ in self.tensors.values()})
        return f"sidecar GGUF {os.path.basename(self.path)} ({len(self.tensors)} tensors, {'/'.join(types)})"

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


_GGML_NAMES = {F32: "F32", F16: "F16", Q8_0: "Q8_0", BF16: "BF16"}
_ST_TYPES = {"F32": (F32, 4), "F16": (F16, 2), "BF16": (BF16, 2)}


class SafetensorsWeights:
    """The full-precision tower from safetensors: ONE file (e.g. the tower extracted from the checkpoint, whose
    __metadata__ carries vision_config / image_token_id) or a checkpoint DIRECTORY (model.safetensors.index.json:
    only the shards holding vision.* / aligner.* / image_* are opened, only those byte ranges are read; the config
    comes from its config.json).  BF16 / F16 / F32 tensors (the DeepSeek-V4.1 checkpoints ship the tower in BF16)."""

    def __init__(self, path):
        self.path = path
        self.meta, cfg = {}, None
        if os.path.isdir(path):
            idx = os.path.join(path, "model.safetensors.index.json")
            if os.path.exists(idx):
                wm = json.load(open(idx))["weight_map"]
                files = sorted({s for n, s in wm.items() if n.startswith(VISION_PREFIXES)})
            else:
                files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
            files = [os.path.join(path, f) for f in files]
            if os.path.exists(os.path.join(path, "config.json")):
                cfg = json.load(open(os.path.join(path, "config.json")))
        else:
            files = [path]
        self.fds, self.tensors = [], {}
        for f in files:
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                hdr = json.loads(fh.read(n))
            self.meta.update(hdr.pop("__metadata__", None) or {})
            fd = _open_ro(f)
            self.fds.append(fd)
            for name, e in hdr.items():
                if not name.startswith(VISION_PREFIXES):
                    continue
                if e["dtype"] not in _ST_TYPES:
                    raise ValueError(f"{f}: {name} is {e['dtype']} (expected BF16 / F16 / F32)")
                b0, b1 = e["data_offsets"]
                self.tensors[name] = (fd, 8 + n + b0, b1 - b0, _ST_TYPES[e["dtype"]][0], tuple(e["shape"]), e["dtype"])
        if not self.tensors:
            raise ValueError(f"{path}: no vision.* / aligner.* / image_* tensors")
        self._cfg = cfg

    def names(self):
        return sorted(self.tensors)

    def get(self, name):
        fd, off, nb, ty, shape, _ = self.tensors[name]
        return dequantize(_pread_all(fd, nb, off, name), ty).reshape(shape).astype(np.float32)

    def torch(self, name, tp, dtype):
        return tp.from_numpy(np.ascontiguousarray(self.get(name))).to(dtype)

    @property
    def vision_config(self):
        if self._cfg is not None and "vision_config" in self._cfg:
            return dict(self._cfg["vision_config"])
        return json.loads(self.meta["vision_config"]) if "vision_config" in self.meta else None

    @property
    def image_token_id(self):
        if self._cfg is not None and "image_token_id" in self._cfg:
            return self._cfg["image_token_id"]
        v = self.meta.get("image_token_id")
        return int(v) if v not in (None, "") else None

    def describe(self):
        types = sorted({e[5] for e in self.tensors.values()})
        return f"safetensors {self.path} ({len(self.tensors)} tensors, {'/'.join(types)})"

    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds = []


class MmprojWeights:
    """The tower from an llama.cpp mmproj GGUF (general.architecture "clip", clip.projector_type "deepseek41v", e.g.
    Unsloth's mmproj-BF16.gguf), its names mapped back to the checkpoint's:
      v.patch_embd.{weight [1024,3,14,14] -> [1024,588], bias}   -> vision.patch_embed.proj.*
      v.blk.N.attn_qkv / attn_out                                  -> vision.blocks.N.attn.wqkv / wo
      v.blk.N.ffn_gate ++ ffn_up (rows: gate first, as mlp.forward's chunk(2)) -> vision.blocks.N.mlp.w1.weight
      v.blk.N.ffn_down / ln1 / ln2                                 -> mlp.w2 / norm1 / norm2
      v.post_ln -> vision.norm;  mm.1 / mm.2 -> aligner.w1 / w2;  v.token_embd.img_start / img_end, v.image_newline
    The clip header has no rope_theta / max_image_tokens: vision_config fills V4.1's 10000 / 1024 (the sidecar's
    config, when one is given too, wins and is cross-checked)."""

    _SIMPLE = {"v.post_ln.weight": "vision.norm.weight", "v.patch_embd.bias": "vision.patch_embed.proj.bias",
               "mm.1.weight": "aligner.w1.weight", "mm.1.bias": "aligner.w1.bias",
               "mm.2.weight": "aligner.w2.weight", "mm.2.bias": "aligner.w2.bias",
               "v.token_embd.img_start": "image_start", "v.token_embd.img_end": "image_end",
               "v.image_newline": "image_newline"}
    _BLOCK = {"attn_qkv.weight": "attn.wqkv.weight", "attn_qkv.bias": "attn.wqkv.bias",
              "attn_out.weight": "attn.wo.weight", "attn_out.bias": "attn.wo.bias",
              "ffn_down.weight": "mlp.w2.weight", "ln1.weight": "norm1.weight", "ln2.weight": "norm2.weight"}

    def __init__(self, path):
        self.kv, tensors, self.data_start = read_gguf_header(path)
        self.path = path
        if self.kv.get("clip.projector_type") != "deepseek41v":
            raise ValueError(f"{path}: clip.projector_type {self.kv.get('clip.projector_type')!r}, expected deepseek41v")
        self.src = {n: (s, t, o) for n, s, t, o in tensors}
        self.fd = _open_ro(path)
        self.map = {}                                  # checkpoint name -> list of mmproj names (concatenated on dim 0)
        for n in self.src:
            if n in self._SIMPLE:
                self.map[self._SIMPLE[n]] = [n]
            elif n == "v.patch_embd.weight":
                self.map["vision.patch_embed.proj.weight"] = [n]
            elif n.startswith("v.blk."):
                i, rest = n[len("v.blk."):].split(".", 1)
                if rest in self._BLOCK:
                    self.map[f"vision.blocks.{i}.{self._BLOCK[rest]}"] = [n]
                elif rest == "ffn_gate.weight":
                    self.map[f"vision.blocks.{i}.mlp.w1.weight"] = [n, f"v.blk.{i}.ffn_up.weight"]
                elif rest != "ffn_up.weight":
                    raise ValueError(f"{path}: unmapped tensor {n}")
            else:
                raise ValueError(f"{path}: unmapped tensor {n}")

    def names(self):
        return sorted(self.map)

    def _one(self, n):
        shape, ty, off = self.src[n]
        return dequantize(_pread_all(self.fd, ggml_nbytes(shape, ty), self.data_start + off, n), ty).reshape(shape[::-1])

    def get(self, name):
        parts = [self._one(n) for n in self.map[name]]
        a = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        if name == "vision.patch_embed.proj.weight":
            a = a.reshape(a.shape[0], -1)              # [1024, 3, p, p] conv -> the reference Linear's [1024, 3*p*p]
        return a.astype(np.float32)

    def torch(self, name, tp, dtype):
        return tp.from_numpy(np.ascontiguousarray(self.get(name))).to(dtype)

    @property
    def vision_config(self):
        k = self.kv
        return {"model_type": "deepseek_v41_vision", "num_hidden_layers": int(k["clip.vision.block_count"]),
                "hidden_size": int(k["clip.vision.embedding_length"]),
                "num_attention_heads": int(k["clip.vision.attention.head_count"]),
                "intermediate_size": int(k["clip.vision.feed_forward_length"]),
                "patch_size": int(k["clip.vision.patch_size"]), "rope_theta": 10000,
                "downsample_ratio": int(k["clip.vision.projector.scale_factor"]), "max_image_tokens": 1024,
                "min_pixels": int(k["clip.vision.image_min_pixels"]), "max_wh_ratio": None}

    @property
    def image_token_id(self):
        return None

    def describe(self):
        types = sorted({_GGML_NAMES.get(t, str(t)) for _, t, _ in self.src.values()})
        return f"mmproj GGUF {os.path.basename(self.path)} ({len(self.src)} tensors -> {len(self.map)}, {'/'.join(types)})"

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def open_weights(path):
    """--vision-weights: a safetensors file or checkpoint directory, an mmproj GGUF (clip / deepseek41v), or another
    V4.1 sidecar GGUF (vision.* names)."""
    if os.path.isdir(path) or path.endswith(".safetensors"):
        return SafetensorsWeights(path)
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"GGUF":
        kv, _, _ = read_gguf_header(path)
        return MmprojWeights(path) if kv.get("general.architecture") == "clip" else SidecarWeights(path)
    return SafetensorsWeights(path)


# keys a weights source's vision_config must agree on with the sidecar's (when both are given)
_CFG_KEYS = ("num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size", "patch_size",
             "downsample_ratio", "min_pixels")


# --------------------------------------------------------------------- model
class VisionArgs:
    """The attributes the reference `vision.py` / `image_processor.py` read (config.json keys)."""

    def __init__(self, cfg, dim, max_n_token=None):
        self.vision_n_layers = int(cfg["num_hidden_layers"])
        self.vision_dim = int(cfg["hidden_size"])
        self.vision_n_heads = int(cfg["num_attention_heads"])
        self.vision_inter_dim = int(cfg["intermediate_size"])
        self.vision_patch_size = int(cfg["patch_size"])
        self.vision_rope_theta = float(cfg["rope_theta"])
        self.vision_downsample_ratio = int(cfg["downsample_ratio"])
        self.vision_max_n_token = int(max_n_token or cfg.get("max_image_tokens", 1024))
        self.vision_min_pixels = int(cfg["min_pixels"])
        self.vision_max_wh_ratio = cfg.get("max_wh_ratio")
        self.dim = int(dim)


def _load_reference(reference_dir):
    if os.path.abspath(reference_dir) not in map(os.path.abspath, sys.path):
        sys.path.insert(0, os.path.abspath(reference_dir))
    import image_processor
    import vision
    return vision.ViT, vision.Aligner, image_processor.load_image, image_processor.num_image_tokens


def fast_forwards(vit):
    """Two runtime adaptations of the reference ViT (`vision.py`), the math unchanged, bound to THIS instance only
    (the reference classes stay as written - the gate compares against them):

    * Attention.forward hands scaled_dot_product_attention CONTIGUOUS [1, H, N, d] tensors.  The reference passes
      transposed views, which sends PyTorch's CPU SDPA to its math fallback (an N x N score matrix per head; on the
      GPU a 9000-patch image would need 16 x 9000^2 floats = 5 GiB of scores), and keeps the CUDA flash /
      memory-efficient kernels out.
    * ViT.forward moves the cached 2-D RoPE tables to the patches' device (the reference builds them on the CPU, so a
      CUDA encoder failed in the first block)."""
    import types
    vision = sys.modules[type(vit).__module__]
    F = torch.nn.functional

    def attention_forward(self, x, cos, sin):
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = vision.apply_rotary(q, cos, sin).transpose(0, 1).contiguous()
        k = vision.apply_rotary(k, cos, sin).transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        o = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
        return self.wo(o.transpose(0, 1).reshape(n, -1))

    def vit_forward(self, patches, n_h, n_w):
        x = self.patch_embed(patches)
        cos, sin = vision.get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    for block in vit.blocks:
        block.attn.forward = types.MethodType(attention_forward, block.attn)
    vit.forward = types.MethodType(vit_forward, vit)
    return vit


def largest_grid(max_n_token, downsample):
    """The squarest LLM grid (h, w) with num_image_tokens(h, w) <= max_n_token, and its ViT patch grid - the
    warm-up shape (the activation pool of the largest image the encoder will see)."""
    best = (1, 1)
    for h in range(1, max_n_token):
        w = (max_n_token - 2) // h - 1
        if w < 1:
            break
        if h * w > best[0] * best[1] or (h * w == best[0] * best[1] and abs(h - w) < abs(best[0] - best[1])):
            best = (h, w)
    h, w = best
    return h, w, h * downsample, w * downsample


def num_image_tokens(n_llm_h, n_llm_w):
    return n_llm_h * (n_llm_w + 1) + 2


def sve_grid(n, n_llm_h, n_llm_w):
    """(nx, ny) for the SVE1 record with nx*ny == n.  The reading-order rectangle n_llm_w+1 x n_llm_h
    holds the body (n_llm_h*(n_llm_w+1) rows) but not the 2 delimiter rows, so it is exact only when
    there are none; otherwise the span is emitted as one flat row in reading order."""
    if (n_llm_w + 1) * n_llm_h == n:
        return n_llm_w + 1, n_llm_h
    return n, 1


def build_span(embeds, n_llm_h, n_llm_w, image_start, image_newline, image_end):
    """model.py merge_image_embeddings: [START] + ([IMAGE]*w + [NEWLINE])*h + [END], IMAGE rows in
    reading order.  `embeds` is [n_llm_h*n_llm_w, dim]."""
    w = n_llm_w
    rows = [image_start]
    for r in range(n_llm_h):
        rows.extend(embeds[r * w:(r + 1) * w])
        rows.append(image_newline)
    rows.append(image_end)
    span = torch.stack(rows)
    assert span.shape[0] == num_image_tokens(n_llm_h, n_llm_w), span.shape
    return span


class Encoder:
    def __init__(self, sidecar, reference_dir=DEFAULT_REFERENCE, device="cpu", threads=0, dtype="f32",
                 max_n_token=None, fast_attention=True, offload=False, tf32=False, weights=None):
        """sidecar: the V4.1 sidecar GGUF (vision config + its own Q8_0 tower), may be None when `weights` carries a
        config.  weights: optional full-precision tower (open_weights: safetensors file / checkpoint dir / mmproj
        GGUF) that REPLACES the sidecar's tensors; the sidecar's vision_config then stays the config, cross-checked."""
        if threads and threads > 0:
            torch.set_num_threads(int(threads))
        if device != "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # never open a context (strata_vision.cpp:91-97)
        self.torch = torch
        self.dtype = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[dtype]

        ViT, Aligner, load_image, _ = _load_reference(reference_dir)
        self.load_image, self.Aligner, self.ViT = load_image, Aligner, ViT

        side = SidecarWeights(sidecar) if sidecar else None
        w = open_weights(weights) if weights else side
        if w is None:
            raise ValueError("no vision weights: give the sidecar (--mmproj) and/or --vision-weights")
        cfg = side.vision_config if side is not None else w.vision_config
        if cfg is None:
            raise ValueError(f"{w.describe()}: no vision_config; pass the sidecar (--mmproj) as well")
        if side is not None and w is not side and w.vision_config is not None:
            bad = {k: (cfg.get(k), w.vision_config.get(k)) for k in _CFG_KEYS if cfg.get(k) != w.vision_config.get(k)}
            if bad:
                raise ValueError(f"--vision-weights {w.path} is not this sidecar's tower: config differs {bad}")
        self.weights_desc = w.describe()
        tile = w.get("image_start")
        self.args = VisionArgs(cfg, dim=int(tile.shape[0]), max_n_token=max_n_token)
        self.n_embd = self.args.dim
        tid = side.image_token_id if side is not None else w.image_token_id
        self.image_token_id = int(tid or 0)

        self.vit = ViT(self.args).eval()
        if fast_attention:
            fast_forwards(self.vit)
        self.aligner = Aligner(self.args).eval()
        self.image_start = torch.from_numpy(np.ascontiguousarray(w.get("image_start")))
        self.image_end = torch.from_numpy(np.ascontiguousarray(w.get("image_end")))
        self.image_newline = torch.from_numpy(np.ascontiguousarray(w.get("image_newline")))

        vit_sd, al_sd = {}, {}
        for name in w.names():
            if name.startswith("vision."):
                vit_sd[name[len("vision."):]] = w.torch(name, torch, self.dtype)
            elif name.startswith("aligner."):
                al_sd[name[len("aligner."):]] = w.torch(name, torch, self.dtype)
        self.vit.load_state_dict(vit_sd, strict=True)
        self.aligner.load_state_dict(al_sd, strict=True)
        for src in {id(x): x for x in (side, w) if x is not None}.values():
            src.close()
        # load_state_dict copies INTO the float32 parameters: the modules themselves must take the dtype (bf16 / f16
        # patches on f32 weights fail in the first Linear; on the CPU "bf16" was f32 compute on bf16-rounded weights)
        self.vit, self.aligner = self.vit.to(self.dtype), self.aligner.to(self.dtype)
        self.device = device
        self.offload = bool(offload) and device == "cuda"
        self.host = None
        self.load_ms = 0.0
        if device == "cuda":                                # only if explicitly asked; tests stay on the CPU
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            for n in ("image_start", "image_end", "image_newline"):
                setattr(self, n, getattr(self, n).to("cuda"))
            if self.offload:                                # weights in pinned host RAM, on the card per ENC only
                self.host = {"vit": {k: v.detach().pin_memory() for k, v in self.vit.state_dict().items()},
                             "aligner": {k: v.detach().pin_memory() for k, v in self.aligner.state_dict().items()}}
                self.vit, self.aligner = self.vit.to("meta"), self.aligner.to("meta")
            else:
                self.vit, self.aligner = self.vit.to("cuda"), self.aligner.to("cuda")

    def weight_bytes(self):
        if self.host is not None:
            return sum(t.numel() * t.element_size() for d in self.host.values() for t in d.values())
        return sum(t.numel() * t.element_size() for m in (self.vit, self.aligner) for t in m.state_dict().values())

    def _forward(self, patches, n_vit_h, n_vit_w):
        torch = self.torch
        if self.host is None:
            return self.aligner(self.vit(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)
        # --offload: this ENC's copy of the weights, copied from pinned RAM and dropped after
        from torch.func import functional_call
        t0 = time.time()
        dev = {name: {k: v.to(self.device, non_blocking=True) for k, v in d.items()} for name, d in self.host.items()}
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.load_ms = (time.time() - t0) * 1e3
        x = functional_call(self.vit, dev["vit"], (patches, n_vit_h, n_vit_w))
        return functional_call(self.aligner, dev["aligner"], (x, n_vit_h, n_vit_w))

    def warmup(self, max_n_token=None):
        """One encode of the largest grid (synthetic patches): the CUDA kernels load and torch's pool grows to the
        largest image's activations BEFORE the server starts the engine.  -> (seconds, n_llm_h, n_llm_w)."""
        torch = self.torch
        h, w, vh, vw = largest_grid(int(max_n_token or self.args.vision_max_n_token), self.args.vision_downsample_ratio)
        p = self.args.vision_patch_size
        patches = torch.zeros(vh * vw, 3, p, p, dtype=self.dtype, device=self.device)
        t0 = time.time()
        with torch.inference_mode():
            self._forward(patches, vh, vw)
        if self.device == "cuda":
            torch.cuda.synchronize()
            if self.offload:
                torch.cuda.empty_cache()
        return time.time() - t0, h, w

    def vram(self):
        """-> (reserved, peak reserved) bytes of torch's pool on the card (0, 0 on the CPU)."""
        if self.device != "cuda":
            return 0, 0
        return self.torch.cuda.memory_reserved(), self.torch.cuda.max_memory_reserved()

    def encode(self, image_bytes):
        """-> (span [n, n_embd] float32 CPU tensor, n_llm_h, n_llm_w, n_vit_h, n_vit_w)."""
        torch = self.torch
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self.load_image({"data": image_bytes}, self.args)
        patches = patches.to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            embeds = self._forward(patches, n_vit_h, n_vit_w)
            span = build_span(embeds, n_llm_h, n_llm_w, self.image_start.to(self.dtype),
                              self.image_newline.to(self.dtype), self.image_end.to(self.dtype))
            span = span.float().cpu()
        if self.offload:
            torch.cuda.empty_cache()                        # the weights' copy and the pool go back to the card
        return span, n_llm_h, n_llm_w, n_vit_h, n_vit_w


# --------------------------------------------------------------------- SVE1 file
def write_sve(path, span, nx, ny):
    """span: [n, n_embd] float32 CPU.  Header + float32 rows (generate.cpp:6317-6331)."""
    a = np.ascontiguousarray(span.detach().cpu().float().numpy(), dtype="<f4")
    n, n_embd = a.shape
    assert nx * ny == n, f"the engine requires nx*ny == n ({nx}*{ny} != {n})"
    hdr = struct.pack("<5i", SVE1, n, nx, ny, n_embd)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(hdr)
        a.tofile(f)
    os.replace(tmp, path)
    return n, n_embd


def read_sve(path):
    """The engine's reader: -> (rows [n, n_embd] float32, nx, ny) for the first record."""
    with open(path, "rb") as f:
        hdr = f.read(20)
        if len(hdr) != 20:
            raise ValueError("short SVE1 header")
        magic, n, nx, ny, n_embd = struct.unpack("<5i", hdr)
        if magic != SVE1 or nx < 1 or ny < 1 or nx * ny != n:
            raise ValueError("bad embeddings file")
        rows = np.frombuffer(f.read(n * n_embd * 4), "<f4").reshape(n, n_embd)
    return rows, nx, ny


# --------------------------------------------------------------------- resident process
def serve(enc, args):
    n_embd = enc.n_embd
    if enc.device == "cuda" and not getattr(args, "no_warmup", False):
        s, h, w = enc.warmup(getattr(args, "warmup_tokens", 0) or None)
        r, pk = enc.vram()
        log(f"ds41-vision: warm-up {h}x{w} grid ({num_image_tokens(h, w)} tokens) in {s:.2f} s; torch pool "
            f"{r / 2**20:.0f} MiB (peak {pk / 2**20:.0f}), weights {enc.weight_bytes() / 2**20:.0f} MiB"
            f"{' in pinned RAM (--offload)' if enc.offload else ''}")
    print(f"READY {n_embd}", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line == "QUIT":
            break
        if not line.startswith("ENC "):
            print("ERR expected: ENC <image> <output>", flush=True)
            continue
        rest = line[4:]
        cut = rest.rfind(" ")                       # strata_vision.cpp:46-54 (paths may contain spaces)
        if cut <= 0:
            print("ERR expected: ENC <image> <output>", flush=True)
            continue
        img, out = rest[:cut], rest[cut + 1:]
        t0 = time.time()
        try:
            with open(img, "rb") as f:
                data = f.read()
            span, n_llm_h, n_llm_w, _, _ = enc.encode(data)
            n = int(span.shape[0])
            nx, ny = sve_grid(n, n_llm_h, n_llm_w)
            write_sve(out, span, nx, ny)
            ms = (time.time() - t0) * 1e3
            print(f"OK {n} {nx} {ny} {ms:.0f}", flush=True)
            if enc.device == "cuda":
                r, pk = enc.vram()
                log(f"ds41-vision: ENC {n} tokens in {ms:.0f} ms"
                    f"{f' (weights in {enc.load_ms:.0f} ms)' if enc.offload else ''}; torch pool {r / 2**20:.0f} MiB "
                    f"(peak {pk / 2**20:.0f})")
        except Exception as e:                      # noqa: BLE001 - the protocol wants a line, always
            print(f"ERR {type(e).__name__}: {e}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], add_help=True)
    ap.add_argument("--mmproj", help="the vision sidecar GGUF (strata-vision's mmproj)")
    ap.add_argument("--sidecar", help="alias of --mmproj")
    ap.add_argument("--vision-weights", default=None,
                    help="full-precision tower replacing the sidecar's Q8_0 one: a safetensors file (the tower "
                         "extracted from the checkpoint), a checkpoint directory (model.safetensors.index.json), or "
                         "an mmproj GGUF (clip / deepseek41v); the sidecar still gives the config")
    ap.add_argument("--model", help="the text model (accepted for argv compatibility; not read)")
    ap.add_argument("--gpu", action="store_true", help="use the GPU (accepted; --device cuda)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="cap on image tokens (image_max_tokens)")
    ap.add_argument("--min-tokens", type=int, default=0, help="accepted for compatibility (unused)")
    ap.add_argument("--flash-attn", default=None, help="accepted for compatibility (unused)")
    ap.add_argument("--reference-dir", default=DEFAULT_REFERENCE, help="DeepSeek's reference python dir")
    ap.add_argument("--dtype", choices=["f32", "bf16", "f16"], default="f32")
    ap.add_argument("--ref-attention", action="store_true",
                    help="the reference Attention.forward as written (strided SDPA inputs: the slow CPU math path)")
    ap.add_argument("--offload", action="store_true",
                    help="cuda: weights in pinned host RAM, copied to the card per image (see the module notes)")
    ap.add_argument("--tf32", action="store_true", help="cuda: TF32 tensor cores for f32 matmuls")
    ap.add_argument("--warmup-tokens", type=int, default=0,
                    help="cuda: the warm-up image size in LLM tokens (default --max-tokens / the model's 1024)")
    ap.add_argument("--no-warmup", action="store_true", help="cuda: no warm-up before READY")
    a = ap.parse_args(argv)

    sidecar = a.sidecar or a.mmproj or (None if a.vision_weights else DEFAULT_SIDECAR)
    device = a.device or ("cuda" if a.gpu else "cpu")
    enc = Encoder(sidecar, reference_dir=a.reference_dir, device=device, threads=a.threads,
                  dtype=a.dtype, max_n_token=a.max_tokens or None, fast_attention=not a.ref_attention,
                  offload=a.offload, tf32=a.tf32, weights=a.vision_weights)
    log(f"ds41-vision: {len(enc.vit.blocks)} vit layers, n_embd {enc.n_embd}, image_token_id "
        f"{enc.image_token_id}, dtype {a.dtype}, device {device}, threads {enc.torch.get_num_threads()}; "
        f"weights: {enc.weights_desc}")
    serve(enc, a)


if __name__ == "__main__":
    main()
