"""ds41_vision.py - a resident image encoder for DeepSeek-V4.1-Flash, a drop-in for Strata's
`strata-vision` process.

The process it replaces is `tools/vision/strata_vision.cpp` (PolyStrata; the DeepSeek/Strata fork is
identical), driven by `serve/server.py` (`class Vision`, ~line 1262) and consumed by the engine's
GENI image path (`src/program/generate.cpp`, ~line 5752 and 6313).  Same argv shape, same stdin/stdout
line protocol, same output-file format.

PROTOCOL (strata_vision.cpp:12-16, 184-233)
  argv   : --mmproj <sidecar.gguf> --model <text model .gguf> [--gpu] [--threads N] [--max-tokens N]
           [--min-tokens N] [--flash-attn on|off|auto]        (server.py:1268 builds exactly this)
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
  * Grid: DeepSeek's span is n_llm_h rows of (n_llm_w+1) plus the 2 delimiters, so nx*ny == n cannot
    be the reading-order rectangle (that would need n = n_llm_h*(n_llm_w+1) = n-2).  We emit the flat
    reading order nx = n, ny = 1. V4.1 uses sequential 1-D text positions for the whole span;
    nx/ny do not change the engine's RoPE positions.
  * No mtmd flash-attn / min-tokens knobs (accepted and ignored); no GPU warm-up.
  * The server already caches by image hash (server.py:1363-1388), so this process keeps no cache.
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


class SidecarWeights:
    """Reads only the vision.* / aligner.* / image_* tensors of the sidecar GGUF, by offset."""

    def __init__(self, path):
        self.kv, tensors, self.data_start = read_gguf_header(path)
        self.path = path
        self.tensors = {n: (s, t, o) for n, s, t, o in tensors if n.startswith(VISION_PREFIXES)}
        if not self.tensors:
            raise ValueError(f"{path}: no vision.* / aligner.* / image_* tensors")
        self.fd = os.open(path, os.O_RDONLY)

    def names(self):
        return sorted(self.tensors)

    def get(self, name):
        """-> float32 numpy array in the checkpoint's own HF orientation (C-order over ne0, so
        [out, in] for a Linear weight -- the reverse of the ggml header shape)."""
        shape, ty, off = self.tensors[name]
        nbytes = ggml_nbytes(shape, ty)
        raw = bytearray()
        while len(raw) < nbytes:
            b = os.pread(self.fd, nbytes - len(raw), self.data_start + off + len(raw))
            if not b:
                raise OSError(f"{name}: short read")
            raw += b
        return dequantize(bytes(raw), ty).reshape(shape[::-1]).astype(np.float32)

    def torch(self, name, tp, dtype):
        return tp.from_numpy(np.ascontiguousarray(self.get(name))).to(dtype)


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
                 max_n_token=None):
        if threads and threads > 0:
            torch.set_num_threads(int(threads))
        if device != "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # never open a context (strata_vision.cpp:91-97)
        self.torch = torch
        self.dtype = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[dtype]

        ViT, Aligner, load_image, _ = _load_reference(reference_dir)
        self.load_image, self.Aligner, self.ViT = load_image, Aligner, ViT

        w = SidecarWeights(sidecar)
        cfg = json.loads(w.kv["deepseek41.sidecar.vision_config"])
        tile = w.get("image_start")
        self.args = VisionArgs(cfg, dim=int(tile.shape[0]), max_n_token=max_n_token)
        self.n_embd = self.args.dim
        self.image_token_id = int(w.kv.get("deepseek41.sidecar.image_token_id", 0))

        self.vit = ViT(self.args).eval()
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
        if device == "cuda":                                # only if explicitly asked; tests stay on the CPU
            self.vit, self.aligner = self.vit.to("cuda"), self.aligner.to("cuda")
            for n in ("image_start", "image_end", "image_newline"):
                setattr(self, n, getattr(self, n).to("cuda"))
        self.device = device

    def encode(self, image_bytes):
        """-> (span [n, n_embd] float32 CPU tensor, n_llm_h, n_llm_w, n_vit_h, n_vit_w)."""
        torch = self.torch
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self.load_image({"data": image_bytes}, self.args)
        patches = patches.to(self.dtype)
        with torch.inference_mode():
            embeds = self.aligner(self.vit(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)
        span = build_span(embeds, n_llm_h, n_llm_w, self.image_start.to(self.dtype),
                          self.image_newline.to(self.dtype), self.image_end.to(self.dtype))
        return span.float().cpu(), n_llm_h, n_llm_w, n_vit_h, n_vit_w


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
            print(f"OK {n} {nx} {ny} {(time.time() - t0) * 1e3:.0f}", flush=True)
        except Exception as e:                      # noqa: BLE001 - the protocol wants a line, always
            print(f"ERR {type(e).__name__}: {e}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], add_help=True)
    ap.add_argument("--mmproj", help="the vision sidecar GGUF (strata-vision's mmproj)")
    ap.add_argument("--sidecar", help="alias of --mmproj")
    ap.add_argument("--model", help="the text model (accepted for argv compatibility; not read)")
    ap.add_argument("--gpu", action="store_true", help="use the GPU (accepted; --device cuda)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="cap on image tokens (image_max_tokens)")
    ap.add_argument("--min-tokens", type=int, default=0, help="accepted for compatibility (unused)")
    ap.add_argument("--flash-attn", default=None, help="accepted for compatibility (unused)")
    ap.add_argument("--reference-dir", default=DEFAULT_REFERENCE, help="DeepSeek's reference python dir")
    ap.add_argument("--dtype", choices=["f32", "bf16", "f16"], default="f32")
    a = ap.parse_args(argv)

    sidecar = a.sidecar or a.mmproj or DEFAULT_SIDECAR
    device = a.device or ("cuda" if a.gpu else "cpu")
    enc = Encoder(sidecar, reference_dir=a.reference_dir, device=device, threads=a.threads,
                  dtype=a.dtype, max_n_token=a.max_tokens or None)
    log(f"ds41-vision: {len(enc.vit.blocks)} vit layers, n_embd {enc.n_embd}, image_token_id "
        f"{enc.image_token_id}, dtype {a.dtype}, device {device}, threads {enc.torch.get_num_threads()}")
    serve(enc, a)


if __name__ == "__main__":
    main()
