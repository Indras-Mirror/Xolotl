#!/usr/bin/env python3
"""Export the final shim's actual prepared image/video inputs without LLM inference.

Runs the configured optional mtmd helper. The caller owns GPU reservation, when
backend=cuda. A tiny engine descriptor supplies context/capabilities only; no
model executable is started. The same Service.prepare path handles HTTP input.

The result contains the original request, exact helper prompt, prepared SVE1,
linear token/visual mapping and F32 embeddings for the separate vision oracle.
"""
from __future__ import annotations
import argparse
from array import array
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import threading
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools/m31/serve")]
import serve_m31 as M


def write_array(path, kind, values):
    data = array(kind, values)
    if data.itemsize != 4:
        raise RuntimeError("requires 32-bit Python arrays")
    if sys.byteorder != "little":
        data.byteswap()
    Path(path).write_bytes(data.tobytes())


def export_prepared(ids, sve, ipc, out):
    """Authenticate the shim's SVE rows against its helper's chunk metadata."""
    ipc = Path(ipc)
    manifest = json.loads(ipc.read_text())
    if manifest.get("format") != "m31-vision-ipc-v1":
        raise ValueError("unexpected helper IPC format")
    expected = []
    for chunk in manifest["chunks"]:
        expected.extend(chunk["tokens"] if chunk["type"] == "text" else [-1] * chunk["rows"])
    if list(ids) != expected or len(ids) != manifest["rows"]:
        raise ValueError("final Service.prepare IDs differ from helper's linear chunks")
    mapping, visual = [], 0
    for token in ids:
        mapping.append(visual if token == -1 else -1)
        visual += token == -1
    if visual != manifest["visual_rows"]:
        raise ValueError("prepared visual count differs from helper")
    out = Path(out)
    out.mkdir(exist_ok=False)
    with Path(sve).open("rb") as source, (out / "visual.f32").open("xb") as target:
        for chunk in manifest["chunks"]:
            if chunk["type"] != "image":
                continue
            header = source.read(20)
            if len(header) != 20 or struct.unpack("<5i", header) != (0x31455653, chunk["rows"], chunk["rows"], 1, 6144):
                raise ValueError("final SVE1 record has wrong linear geometry")
            count = chunk["rows"] * 6144 * 4
            data = source.read(count)
            if len(data) != count or data != (ipc.parent / chunk["embeddings"]).read_bytes():
                raise ValueError("final SVE1 embeddings differ from actual helper output")
            target.write(data)
        if source.read(1):
            raise ValueError("extra SVE1 bytes")
    write_array(out / "text_ids.i32", "i", ids)
    write_array(out / "visual_rows.i32", "i", mapping)
    write_array(out / "positions.i32", "i", range(len(ids)))
    manifest.update(format="m31-vision-batch-v1", source_manifest=str(ipc.resolve()),
                    row_contract="text_ids>=0 selects token embedding; -1 selects visual_rows in visual.f32",
                    llm_batches=1, native_engine_validated=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out / "manifest.json"


def assemble_ipc(ipc, out):
    """Independently assemble the reference helper's text/visual arrays."""
    ipc = Path(ipc)
    manifest = json.loads(ipc.read_text())
    if manifest.get("format") != "m31-vision-ipc-v1" or manifest.get("n_embd") != 6144:
        raise ValueError("unsupported independent IPC")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    ids, mapping, visual = [], [], 0
    with (out / "visual.f32").open("xb") as target:
        for index, chunk in enumerate(manifest["chunks"]):
            count = chunk["rows"]
            if (chunk["index"] != index or chunk["row"] != len(ids) or chunk["position"] != len(ids)
                    or chunk["n_pos"] != count or type(count) is not int or count <= 0):
                raise ValueError("invalid independent IPC chunk continuity")
            if chunk["type"] == "text":
                if len(chunk["tokens"]) != count:
                    raise ValueError("invalid independent text chunk")
                ids.extend(chunk["tokens"])
                mapping.extend([-1] * count)
            elif chunk["type"] == "image":
                if chunk["visual_row"] != visual:
                    raise ValueError("invalid independent visual mapping")
                payload = (ipc.parent / chunk["embeddings"]).read_bytes()
                if len(payload) != count * 6144 * 4:
                    raise ValueError("invalid independent embedding length")
                target.write(payload)
                ids.extend([-1] * count)
                mapping.extend(range(visual, visual + count))
                visual += count
            else:
                raise ValueError("unsupported independent chunk type")
    if len(ids) != manifest["rows"] or visual != manifest["visual_rows"]:
        raise ValueError("invalid independent total row count")
    write_array(out / "text_ids.i32", "i", ids)
    write_array(out / "visual_rows.i32", "i", mapping)
    write_array(out / "positions.i32", "i", range(len(ids)))
    manifest.update(format="m31-vision-batch-v1", source_manifest=str(ipc.resolve()),
        row_contract="text_ids>=0 selects token embedding; -1 selects visual_rows in visual.f32",
        llm_batches=1, native_engine_validated=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"mixed_manifest": str(out / "manifest.json"), "rows": len(ids), "visual_rows": visual}


def prepare(request, cfg, tokenizer_dir, out, context=40000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    M.install()
    tokdir = Path(tokenizer_dir)
    vocabulary = json.loads((tokdir / "vocab.json").read_text())
    tokens = [None] * len(vocabulary)
    for token, index in vocabulary.items():
        tokens[index] = token
    metadata = json.loads((tokdir / "tokenizer.json").read_text())
    tokenizer = M.M31Tokenizer(tokens, (tokdir / "merges.txt").read_text().splitlines(),
        json.loads((tokdir / "token_type.json").read_text()), special_ids=list(metadata["special_ids"].values()))
    template = M.M31Template(tokdir / "chat_template.jinja")
    options = dict(cfg, cache_dir=str(out / "helper"), embedding_cache_mb=0)
    vision = M.M31Vision(options)
    descriptor = SimpleNamespace(max_context=context, known_ctx=context, info={"vision": 1}, batch=0)
    service = M.M31Service(descriptor, tokenizer, template, vision=vision)
    capture = {}
    original = vision.encode_prompt
    def encode(prompt, media, fps, cancel=None):
        snapshots = out / "media"
        snapshots.mkdir(exist_ok=True)
        retained = []
        for index, path in enumerate(media):
            retained_path = snapshots / (str(index) + "-" + Path(path).name)
            shutil.copyfile(path, retained_path)
            retained.append(str(retained_path))
        capture.update(prompt=prompt, media=retained, video_fps=fps)
        result = original(prompt, media, fps, cancel)
        # Service.prepare releases its request-owned raw IPC after conversion.
        # Preserve it before returning from the same helper call; no input or
        # runtime result is replaced by this diagnostic snapshot.
        saved = out / "captured-ipc"
        shutil.copytree(Path(result).parent, saved)
        capture["ipc"] = str(saved / Path(result).name)
        return result
    vision.encode_prompt = encode
    try:
        messages, tools, kwargs = M.F.openai_to_messages(request)
        ids, thinking, maximum = service.prepare(messages, tools, kwargs,
            request.get("max_tokens", 1200), req=copy.deepcopy(request), cancel=threading.Event())
        if "ipc" not in capture or not getattr(service.embeddings, "path", None):
            raise ValueError("request did not exercise final vision preparation")
        shutil.copyfile(service.embeddings.path, out / "prepared.sve")
        (out / "prompt.txt").write_text(capture["prompt"])
        (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        mixed = export_prepared(ids, out / "prepared.sve", capture["ipc"], out / "mixed")
        spec = {"model": options["model"], "mmproj": options["mmproj"], "backend": options["backend"],
            "threads": options.get("threads", 16), "encode": True, "encoder_mode": "single", "add_special": False,
            "prompt": capture["prompt"], "media": capture["media"], "video_fps": capture["video_fps"],
            "timestamp_interval_ms": options.get("timestamp_interval_ms", 5000)}
        (out / "independent-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        receipt = {"format": "m31-final-prepared-input-v1", "llm_inference": False,
            "rows": len(ids), "visual_rows": sum(token == -1 for token in ids), "mixed_manifest": str(mixed),
            "actual_helper_manifest": capture["ipc"], "thinking": thinking, "max_new": maximum,
            "vision_config": options, "encoder_stats": vision.client.last_stats,
            "sve_sha256": hashlib.sha256((out / "prepared.sve").read_bytes()).hexdigest()}
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    finally:
        service.drop_embeddings()
        vision.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path)
    p.add_argument("--ipc", type=Path, help="assemble an already encoded independent helper IPC instead")
    p.add_argument("--vision-config", type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--context", type=int, default=40000)
    a = p.parse_args()
    if a.ipc:
        result = assemble_ipc(a.ipc, a.out)
    else:
        if not all((a.request, a.vision_config, a.tokenizer)):
            p.error("--request, --vision-config and --tokenizer are required without --ipc")
        result = prepare(json.loads(a.request.read_text()), json.loads(a.vision_config.read_text()),
                         a.tokenizer, a.out, a.context)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
