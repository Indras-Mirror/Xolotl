"""test_ds41_vision.py - the gate for tools/v41/vision/ds41_vision.py.

    python3 tools/v41/vision/test_ds41_vision.py            # CPU, 4 torch threads (run it under nice -n 10)
    DS41_VISION_WEIGHTS=<tower.safetensors | checkpoint dir | mmproj.gguf> ...   # the encoder on the full tower
    DS41_TEST_DEVICE=cuda ...                                 # the encoder on the GPU (the reference stays on the CPU)

1. Reference = DeepSeek's own ViT + Aligner (reference `vision.py`) with the ORIGINAL checkpoint weights
   (the FP8/BF16 shards holding vision.*/aligner.*/image_*), the span assembled the way
   model.py:merge_image_embeddings does.  Encoder = the sidecar weights, or DS41_VISION_WEIGHTS (the encoder's
   --vision-weights) when set.  Per-row cosine over the whole span: min >= 0.999.  (The sidecar's Q8_0 tower
   alone is ~0.996 on the 300x1200 image; the full tower passes.)
2. The same comparison with the reference built from the encoder's OWN weights (the dequantized sidecar, or the
   --vision-weights tensors): the forward path and the span assembly must agree to <= 1e-2 (bf16 level).
   With DS41_VISION_WEIGHTS the max |diff| of those tensors against the checkpoint's is printed too (0 for BF16).
3. Three synthetic PIL images (640x480 text+shapes, 300x1200 tall, 50x50 tiny): every row count equals
   image_processor.num_image_tokens(plan_image_grid(...)).
4. Protocol smoke: spawn ds41_vision.py exactly as serve/server.py's `class Vision` would, encode two
   images, and read the files the engine's GENI path reads (generate.cpp:6313-6331).

No image is ever read from disk as a fixture or displayed; images exist only in memory (a temp file is
written solely because the ENC protocol takes a path).  By default the GPU is untouched: the encoder forces
CUDA_VISIBLE_DEVICES=-1 and the test never calls torch.cuda.  DS41_TEST_DEVICE=cuda puts the encoder (in-process
and the protocol-smoke subprocess, --device cuda) on the GPU; there the subprocess rows must match the in-process
ones to 1e-5 relative instead of bit for bit.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V41 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(V41))
for p in (HERE, V41, os.path.join(REPO, "third_party", "llama.cpp", "gguf-py")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
torch.set_num_threads(4)

from PIL import Image, ImageDraw  # noqa: E402

import ds41_vision as V  # noqa: E402
import v41_gguf as G  # noqa: E402
from gguf import quants as Q  # noqa: E402

SIDECAR = os.environ.get("DS41_SIDECAR", V.DEFAULT_SIDECAR)
REFERENCE = os.environ.get("DS41_REFERENCE", V.DEFAULT_REFERENCE)
CHECKPOINT = os.environ.get("V41_CHECKPOINT", G.MAIN)
WEIGHTS = os.environ.get("DS41_VISION_WEIGHTS") or None
DEVICE = os.environ.get("DS41_TEST_DEVICE", "cpu")

sys.path.insert(0, os.path.abspath(REFERENCE))
import image_processor as IP  # noqa: E402
import vision as REF  # noqa: E402


def ref_args(cfg):
    """The attribute names reference vision.py / image_processor.py read, from the HF checkpoint's
    nested vision_config / text_config."""
    vc, tc = cfg["vision_config"], cfg["text_config"]
    a = types.SimpleNamespace()
    a.vision_n_layers = int(vc["num_hidden_layers"])
    a.vision_dim = int(vc["hidden_size"])
    a.vision_n_heads = int(vc["num_attention_heads"])
    a.vision_inter_dim = int(vc["intermediate_size"])
    a.vision_patch_size = int(vc["patch_size"])
    a.vision_rope_theta = float(vc["rope_theta"])
    a.vision_downsample_ratio = int(vc["downsample_ratio"])
    a.vision_max_n_token = int(vc["max_image_tokens"])
    a.vision_min_pixels = int(vc["min_pixels"])
    a.vision_max_wh_ratio = vc["max_wh_ratio"]
    a.dim = int(tc["hidden_size"])
    return a


# --------------------------------------------------------------------- reference weight sources
def checkpoint_weights():
    """{name: float32 numpy array in HF orientation} straight from the FP8/BF16 checkpoint shards."""
    src = G.Source([CHECKPOINT])
    names = [n for n in src.wm if n.startswith(("vision.", "aligner.", "image_"))]
    return {n: src.f32(n).astype(np.float32) for n in names}


def sidecar_weights():
    """The sidecar's own tensors, loaded by a path independent of ds41_vision.py (v41_gguf + gguf-py)."""
    kv, ts, start = G.read_gguf(SIDECAR)
    out = {}
    with open(SIDECAR, "rb") as f:
        for name, shape, ty, off in ts:
            if not name.startswith(("vision.", "aligner.", "image_")):
                continue
            n = int(np.prod(shape))
            if ty == 0:                                      # F32
                nb, dec = n * 4, None
            elif ty == 8:                                    # Q8_0
                nb, dec = n // 32 * 34, Q.GGMLQuantizationType.Q8_0
            else:
                raise AssertionError(f"{name}: unexpected type {ty}")
            f.seek(start + off)
            raw = f.read(nb)
            a = np.frombuffer(raw, np.float32).copy() if dec is None else \
                np.asarray(Q.dequantize(np.frombuffer(raw, np.uint8), dec), np.float32).reshape(-1)
            out[name] = a.reshape(shape[::-1])
    return kv, out


# --------------------------------------------------------------------- the reference forward
def reference_span(weights, patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w, args):
    """model.py's encode_image + merge_image_embeddings, built by hand from the reference classes."""
    to = lambda n: torch.from_numpy(np.ascontiguousarray(weights[n]))  # noqa: E731
    vit, al = REF.ViT(args).eval(), REF.Aligner(args).eval()
    vit.load_state_dict({k: to("vision." + k) for k in vit.state_dict()}, strict=True)
    al.load_state_dict({k: to("aligner." + k) for k in al.state_dict()}, strict=True)
    patches = patches.to(torch.float32)
    types_ = IP.image_token_types(n_llm_h, n_llm_w)
    with torch.inference_mode():
        embeds = al(vit(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)
    span = torch.empty(types_.numel(), args.dim)
    span[types_ == IP.IMAGE_START] = to("image_start")
    span[types_ == IP.IMAGE_END] = to("image_end")
    span[types_ == IP.IMAGE_NEW_LINE] = to("image_newline")
    span[types_ == IP.IMAGE] = embeds
    return span, embeds


# --------------------------------------------------------------------- synthetic images
def synth_images():
    out = []

    im = Image.new("RGB", (640, 480), (248, 248, 244))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 260, 220], fill=(190, 30, 30))
    d.ellipse([300, 60, 560, 300], fill=(30, 90, 200))
    d.polygon([(360, 430), (300, 330), (420, 330)], fill=(20, 140, 60))
    d.text((60, 350), "HELLO VISION 41", fill=(0, 0, 0))
    d.text((60, 400), "STRATA SVE1", fill=(0, 0, 0))
    out.append(("640x480 text+shapes", im))

    im = Image.new("RGB", (300, 1200), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i in range(6):
        d.rectangle([20, 40 + i * 190, 280, 170 + i * 190],
                    fill=((40 + 35 * i) % 255, (200 - 30 * i) % 255, (90 + 20 * i) % 255))
        d.text((40, 90 + i * 190), f"ROW {i}", fill=(0, 0, 0))
    out.append(("300x1200 tall", im))

    im = Image.new("RGB", (50, 50), (10, 10, 10))
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, 42, 42], fill=(250, 210, 40))
    out.append(("50x50 tiny", im))

    imgs = []
    for label, im in out:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        imgs.append((label, buf.getvalue()))
    return imgs


# --------------------------------------------------------------------- the gate
def main():
    t_start = time.time()
    cfg = json.load(open(os.path.join(CHECKPOINT, "config.json")))
    args = ref_args(cfg)
    print(f"reference dir : {REFERENCE}")
    print(f"checkpoint    : {CHECKPOINT}")
    print(f"sidecar       : {SIDECAR}")
    print(f"weights       : {WEIGHTS or '(the sidecar)'}")
    print(f"device        : {DEVICE}")

    print("\n== item 1: sidecar Q8_0 vs the sidecar's delimiters sanity ==")
    kv, side = sidecar_weights()
    print(f"image_token_id {kv['deepseek41.sidecar.image_token_id']}, {len(side)} vision tensors dequantized")

    t0 = time.time()
    ck = checkpoint_weights()
    print(f"checkpoint vision tensors: {len(ck)} loaded in {time.time() - t0:.1f}s")

    own = side
    if WEIGHTS:
        src = V.open_weights(WEIGHTS)
        own = {n: src.get(n) for n in src.names()}
        src.close()
        assert sorted(own) == sorted(ck), (len(own), len(ck))
        worst = max(float(np.abs(own[n] - ck[n]).max()) for n in ck)
        nz = sum(float(np.abs(own[n] - ck[n]).max()) != 0.0 for n in ck)
        print(f"--vision-weights vs checkpoint: {len(own)} tensors, max |diff| {worst:.3g} ({nz} tensors differ)")

    t0 = time.time()
    enc = V.Encoder(SIDECAR, reference_dir=REFERENCE, device=DEVICE, threads=4, dtype="f32", weights=WEIGHTS)
    print(f"encoder built in {time.time() - t0:.1f}s (n_embd {enc.n_embd}, weights: {enc.weights_desc})")
    if DEVICE == "cuda":
        s, h, w = enc.warmup()
        print(f"cuda warm-up {h}x{w} grid in {s:.2f}s")

    imgs = synth_images()
    results, spans = [], {}
    print("\n== items 1-3: per-image comparison ==")
    for label, data in imgs:
        e0 = time.time()
        span_e, n_llm_h, n_llm_w, n_vit_h, n_vit_w = enc.encode(data)
        enc_s = time.time() - e0
        if DEVICE == "cuda":                                 # a second, warm encode for the timing column
            e0 = time.time()
            span_e2 = enc.encode(data)[0]
            enc_s = time.time() - e0
            assert float((span_e2 - span_e).abs().max()) <= 1e-5 * float(span_e.abs().max()), "cuda encode not stable"
        patches, vh, vw, nh, nw = IP.load_image({"data": data}, args)
        assert (vh, vw, nh, nw) == (n_vit_h, n_vit_w, n_llm_h, n_llm_w)
        n_expect = IP.num_image_tokens(n_llm_h, n_llm_w)
        assert span_e.shape == (n_expect, args.dim), (span_e.shape, n_expect)
        nx, ny = V.sve_grid(n_expect, n_llm_h, n_llm_w)
        assert nx * ny == n_expect

        span_ck, _ = reference_span(ck, patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w, args)
        span_sc, _ = reference_span(own, patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w, args)

        cos = torch.nn.functional.cosine_similarity(span_e, span_ck, dim=1)
        rel_ck = ((span_e - span_ck).norm(dim=1) / span_ck.norm(dim=1).clamp_min(1e-12)).max().item()
        rel_sc = ((span_e - span_sc).norm(dim=1) / span_sc.norm(dim=1).clamp_min(1e-12)).max().item()
        results.append((label, n_llm_h, n_llm_w, n_vit_h, n_vit_w, n_expect, nx, ny,
                        cos.min().item(), cos.mean().item(), rel_ck, rel_sc, enc_s))
        spans[label] = span_e
        print(f"  {label:20s} vit {n_vit_h}x{n_vit_w} llm {n_llm_h}x{n_llm_w} rows {n_expect:4d} "
              f"grid {nx}x{ny} | cos min {cos.min().item():.6f} mean {cos.mean().item():.6f} | "
              f"relerr vs FP8 {rel_ck:.2e} vs own weights {rel_sc:.2e} | {enc_s:.2f}s")

    print("\n== item 4: protocol smoke (subprocess, argv + stdin as server.py's Vision) ==")
    smoke = protocol_smoke(imgs, results, spans)

    print("\n== GATE ==")
    ok = True
    for r in results:
        good = r[8] >= 0.999 and r[11] <= 1e-2
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} {r[0]:20s} cosine>={0.999} (min {r[8]:.6f}), "
              f"own-weights relerr<={1e-2} ({r[11]:.2e})")
    ok &= smoke
    print(f"  {'PASS' if smoke else 'FAIL'} protocol smoke")
    print(f"\ntotal {time.time() - t_start:.1f}s")
    sys.exit(0 if ok else 1)


def protocol_smoke(imgs, results, spans):
    """Spawn the encoder the way `class Vision` does and read the SVE1 files the engine reads."""
    tmp = tempfile.mkdtemp(prefix="ds41-vision-test-")
    argv = [sys.executable, os.path.join(HERE, "ds41_vision.py"),
            "--mmproj", SIDECAR, "--model", "unused.gguf", "--threads", "4",
            "--reference-dir", REFERENCE]
    if WEIGHTS:
        argv += ["--vision-weights", WEIGHTS]
    if DEVICE == "cuda":
        argv += ["--device", "cuda"]
        env = dict(os.environ)
    else:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    ready = proc.stdout.readline().strip()                     # server.py:1297 demands "READY"
    assert ready.startswith("READY"), f"no READY line: {ready!r}"
    print(f"  {ready}")
    outs = []
    for k in (0, 2):
        label, data = imgs[k]
        n_expect, (nx_e, ny_e) = results[k][5], results[k][6:8]
        img = os.path.join(tmp, f"in{k}.png")
        out = os.path.join(tmp, f"out{k}.sve")
        with open(img, "wb") as f:
            f.write(data)
        proc.stdin.write(f"ENC {img} {out}\n")                 # server.py:1376
        proc.stdin.flush()
        line = proc.stdout.readline().strip()
        assert line.startswith("OK"), f"encoder said: {line!r}"
        n, nx, ny = (int(x) for x in line.split()[1:4])
        rows, gx, gy = V.read_sve(out)                         # engine parse (generate.cpp:6319)
        ref_rows = spans[label].numpy()
        dev = float(np.abs(rows - ref_rows).max())
        tol = 0.0 if DEVICE != "cuda" else 1e-5 * float(np.abs(ref_rows).max())
        ok = (n == n_expect and rows.shape == (n_expect, 5120) and (nx, ny) == (gx, gy) == (nx_e, ny_e)
              and dev <= tol)
        print(f"  ENC {label:20s} -> {line} | file {rows.shape} grid {gx}x{gy} | "
              f"rows == in-process: {dev <= tol} (max |diff| {dev:g}{f', tol {tol:.2g}' if tol else ''})")
        outs.append(ok)
    proc.stdin.write("QUIT\n")
    proc.stdin.flush()
    assert proc.wait(timeout=60) == 0, "encoder did not exit cleanly on QUIT"
    err = proc.stderr.read()
    note = [ln for ln in err.splitlines() if ln.startswith("ds41-vision:")]
    if note:
        print("  " + note[-1])
    return all(outs)


if __name__ == "__main__":
    main()
