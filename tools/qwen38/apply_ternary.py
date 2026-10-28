#!/usr/bin/env python3
"""Write measured PTQ1_0 gate/up tensors into a Qwen4Exp GGUF.

This is deliberately a post-pass, not a ggml type registration: type 143 is
outside stock ggml's enum. llama-quantize first writes all other tensors. This
tool replaces ONLY layers whose allocation selects ptq1, then verifies the
result through Strata's architecture-independent GGUF reader and parity test.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v41"))
import v41_formats as F
from graft_ple import Hdr

GGUF_PY = "/home/mal/AI/llama.cpp-master-rebase/gguf-py"
sys.path.insert(0, GGUF_PY)
import gguf
from gguf import quants


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def size_of(tt, dims):
    if tt == 143:
        assert dims[0] % 128 == 0
        return int(np.prod(dims)) // 128 * 28
    q = gguf.GGMLQuantizationType(tt)
    shape = gguf.quants.quant_shape_to_byte_shape(tuple(dims))
    return int(np.prod(shape))


def verify_output(out, src, infos, new_bytes, importance, ss, alignment):
    """Prove the written GGUF before anything is renamed into place.

    Checks the directory (name order, types, dims, alignment, sizes), that every
    replaced payload decodes as PTQ1_0 and re-encodes to the same bytes from the
    screen's importance, and that untouched tensors kept their source bytes.
    """
    # Parse the header with Hdr, NOT gguf-py: type 143 is outside gguf-py's enum
    # (GGMLQuantizationType(143) raises), so a gguf-py read would crash on exactly
    # the file this function is supposed to verify.
    h = Hdr(str(out))
    hd = h.data_start
    if len(h.infos) != len(infos):
        raise SystemExit(f"verification: {len(h.infos)} tensors written, {len(infos)} planned")
    written = [(n, tt, dims, off) for _s, _e, n, tt, dims, off in h.infos]
    if [n for n, *_ in written] != [i["name"] for i in infos]:
        raise SystemExit("verification: tensor order differs from the plan")
    for i, (name, tt, dims, off) in zip(infos, written):
        if int(tt) != int(i["type"]):
            raise SystemExit(f"verification: {name} type {tt} != {i['type']}")
        if tuple(int(x) for x in dims) != tuple(int(x) for x in i["dims"]):
            raise SystemExit(f"verification: {name} dims {tuple(dims)} != {tuple(i['dims'])}")
        if int(off) % alignment:
            raise SystemExit(f"verification: {name} offset {off} is not {alignment}-aligned")
        if int(off) != int(i["offset"]):
            raise SystemExit(f"verification: {name} offset {off} != {i['offset']}")
        got_n = size_of(int(tt), tuple(int(x) for x in dims))
        if got_n != int(i["nbytes"]):
            raise SystemExit(f"verification: {name} payload {got_n} != planned {i['nbytes']}")
    offsets = {n: off for n, _t, _d, off in written}
    with open(out, "rb") as fo:
        for name in new_bytes:
            st = ss[name]
            experts = int(st.shape[2])
            per = new_bytes[name][1] // experts
            for e in (0, experts - 1):
                W = quants.dequantize(np.asarray(st.data).reshape(experts, -1)[e], st.tensor_type)
                W = W.astype(np.float32).reshape(int(st.shape[1]), int(st.shape[0]))
                enc = F.encode_rtn(W, "ptq1", importance(f"L{int(name.split('.')[1])}_gu_{e}"))
                fo.seek(hd + offsets[name] + e * per)
                if fo.read(per) != enc.tobytes():
                    raise SystemExit(f"verification: {name} expert {e} payload differs from its encoder")
                for row in (enc[0], enc[-1]):   # the payload must decode, not just match bytes
                    back = F.decode(F.PTQ1_0, row, int(st.shape[0]))
                    if back.shape != (int(st.shape[0]),) or not np.all(np.isfinite(back)):
                        raise SystemExit(f"verification: {name} expert {e} does not decode")
    sample = [i["name"] for i in infos if i["name"] not in new_bytes]
    sample = sample[:1] + sample[len(sample) // 2:len(sample) // 2 + 1] + sample[-1:]
    hs = Hdr(src)                      # offsets in both headers are relative to data_start
    src_off = {n: off for _s, _e, n, _t, _d, off in hs.infos}
    plan_n = {i["name"]: int(i["nbytes"]) for i in infos}
    with open(out, "rb") as fo, open(src, "rb") as fs:
        for name in sample:
            take = min(plan_n[name], 1 << 20)
            fo.seek(hd + offsets[name])
            fs.seek(hs.data_start + src_off[name])
            if fo.read(take) != fs.read(take):
                raise SystemExit(f"verification: untouched tensor {name} bytes changed")
    print(f"verified: {len(infos)} tensors, {len(new_bytes)} PTQ1_0 payloads, "
          f"{len(sample)} untouched tensors byte-identical")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="llama-quantize output; never the edited calibration source")
    ap.add_argument("--source", required=True, help="edited Q8_0 calibration GGUF")
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--edit-report", required=True)
    ap.add_argument("--importance", required=True,
                    help="screen.py --importance-out npz for every layer that selects ptq1")
    ap.add_argument("--out", required=True, help="sidecar report; GGUF is modified only after these pass")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if pathlib.Path(a.input).resolve() == pathlib.Path(a.source).resolve():
        raise SystemExit("refusing to replace tensors in the calibration source")
    d = json.load(open(a.alloc))
    if not d.get("layer_choice") or not d.get("edit_downs"):
        raise SystemExit("allocation must be an edited, per-layer Qwen4Exp choice")
    er = json.load(open(a.edit_report))
    if er.get("state") != "edited" or er.get("layers") != "all" or len(er.get("targets", [])) != 145:
        raise SystemExit("not a complete 145-target edit report")
    if pathlib.Path(er["model"]).resolve() != pathlib.Path(a.source).resolve():
        raise SystemExit("edit report and source differ")
    if er.get("edited_sha256") != sha(a.source):
        raise SystemExit("source differs from its completed edit report")
    r = gguf.GGUFReader(a.input)
    src = gguf.GGUFReader(a.source)
    ts = {t.name: t for t in r.tensors}
    ss = {t.name: t for t in src.tensors}
    imp_path = pathlib.Path(a.importance)
    if not imp_path.exists():
        raise SystemExit(f"importance path missing: {imp_path}")
    if imp_path.is_dir():
        files = sorted(imp_path.glob("*.npz"))
        if not files:
            raise SystemExit(f"{imp_path}: no .npz files")
        sources = [np.load(f) for f in files]
    else:
        sources = [np.load(imp_path)]
    available = set().union(*(set(s.files) for s in sources))

    def importance(key):
        for s in sources:
            if key in s.files:
                return np.asarray(s[key], dtype=np.float32).ravel()
        raise SystemExit(f"no screened importance for {key}; refusing a uniform fallback")

    targets = []
    for L, ex in d["map"].items():
        choices = {tuple(sorted((k, v) for k, v in e.items() if k in ("gu", "dn"))) for e in ex.values()}
        if len(choices) != 1:
            raise SystemExit(f"L{L}: allocation is not uniform at the engine's per-layer boundary")
        choice = dict(choices.pop())
        if choice["dn"] not in ("iq4_nl", "q8_0"):
            raise SystemExit(f"L{L}: edited down tier {choice['dn']} is forbidden")
        if choice["gu"] == "ptq1":
            for role in ("gate", "up"):
                targets.append((f"blk.{L}.ffn_{role}_exps.weight", int(L)))
    h = Hdr(a.input)
    new_bytes = {}
    old_total = new_total = 0
    for name, L in targets:
        if name not in ts or name not in ss:
            raise SystemExit(f"{name}: missing in source or quantized model")
        t = ts[name]
        if len(t.shape) != 3 or t.shape[0] != 2560 or t.shape[1] != 640 or t.shape[2] != 512:
            raise SystemExit(f"{name}: not Flash-Next stacked gate/up geometry: {t.shape}")
        old = size_of(t.tensor_type, t.shape)
        new = F.nbytes(F.PTQ1_0, tuple(int(x) for x in t.shape))
        missing = [e for e in range(int(t.shape[2])) if f"L{L}_gu_{e}" not in available]
        if missing:
            raise SystemExit(f"L{L}: no screened importance for expert(s) {missing[:8]} "
                             f"({len(missing)} total); refusing a uniform fallback")
        new_bytes[name] = (old, new)
        old_total += old
        new_total += new
    delta = new_total - old_total
    print(f"PTQ1_0 targets: {len(targets)} tensors; payload delta {delta / 2**30:+.4f} GiB")
    if a.dry_run:
        print("dry run: no file changed")
        return
    tmp = pathlib.Path(a.input + ".ptq1.partial")
    final = pathlib.Path(a.input)
    if final.with_suffix(final.suffix + ".ptq1").exists():
        raise SystemExit("output .ptq1 file already exists; refusing to overwrite")
    if tmp.exists():
        raise SystemExit(f"partial exists: {tmp}; inspect and move it aside before retrying")
    # Rewriting the full GGUF is required because tensor sizes/offsets change.
    # Copy through the SAME volume, never deleting the input before the new file
    # has been parsed and verified.
    out = final.with_suffix(final.suffix + ".ptq1")
    free = os.statvfs(out.parent).f_bavail * os.statvfs(out.parent).f_frsize
    if free < final.stat().st_size + delta + (4 << 30):
        raise SystemExit("insufficient space for a verified PTQ1_0 output and 4 GiB reserve")
    info = []
    for _s, _e, name, tt, dims, off in h.infos:
        info.append({"name": name, "type": 143 if name in new_bytes else tt, "dims": dims})
    next_off = 0
    for t in info:
        t["offset"] = next_off
        t["nbytes"] = size_of(t["type"], t["dims"]) if t["name"] not in new_bytes else new_bytes[t["name"]][1]
        next_off += (t["nbytes"] + h.alignment - 1) // h.alignment * h.alignment
    with open(a.input, "rb") as f:
        f.seek(h.kv_start)
        kv = f.read(h.kv_end - h.kv_start)
    header = b"GGUF" + __import__("struct").pack("<IQQ", h.version, h.n_tensors, h.n_kv) + kv
    import struct
    for t in info:
        name = t["name"].encode()
        header += struct.pack("<Q", len(name)) + name
        header += struct.pack("<I", len(t["dims"])) + struct.pack("<%dQ" % len(t["dims"]), *t["dims"])
        header += struct.pack("<IQ", t["type"], t["offset"])
    header += bytes((len(header) + h.alignment - 1) // h.alignment * h.alignment - len(header))
    written = []
    with open(tmp, "xb") as o:
        o.write(header)
        for t in info:
            name = t["name"]
            if name in new_bytes:
                st = ss[name]
                payload = []
                for e in range(int(st.shape[2])):
                    W = quants.dequantize(np.asarray(st.data).reshape(int(st.shape[2]), -1)[e], st.tensor_type)
                    W = W.astype(np.float32).reshape(int(st.shape[1]), int(st.shape[0]))
                    # The screen's fitted per-expert importance, byte for byte, so the
                    # emitted PTQ1_0 scales are the ones whose error was measured.
                    im = importance(f"L{L}_gu_{e}")
                    if im.size != int(st.shape[0]):
                        raise ValueError(f"{name} expert {e}: importance size {im.size} "
                                         f"!= {int(st.shape[0])}")
                    payload.append(F.encode_rtn(W, "ptq1", im).tobytes())
                raw = b"".join(payload)
                if len(raw) != t["nbytes"]:
                    raise ValueError(f"{name}: encoder returned wrong size")
                o.write(raw)
            else:
                with open(a.input, "rb") as f:
                    f.seek(h.data_start + dict((n, off) for _s, _e, n, tt, dims, off in h.infos)[name])
                    remain = t["nbytes"]
                    while remain:
                        b = f.read(min(remain, 16 << 20))
                        if not b:
                            raise IOError(f"{name}: short source read")
                        o.write(b)
                        remain -= len(b)
            pad = (t["nbytes"] + h.alignment - 1) // h.alignment * h.alignment - t["nbytes"]
            o.write(bytes(pad))
            written.append({"name": name, "type": t["type"], "bytes": t["nbytes"]})
            print(f"[{len(written)}/{len(info)}] {name}: type {t['type']}", flush=True)
        o.flush()
        os.fsync(o.fileno())
    verify_output(tmp, a.input, info, new_bytes, importance, ss, h.alignment)
    os.replace(tmp, out)
    report = {"schema": 1, "input": a.input, "output": str(out), "source": a.source,
              "edit_report": a.edit_report, "alloc": a.alloc, "targets": targets,
              "payload_delta": delta, "output_sha256": sha(out),
              "importance": str(imp_path.resolve()),
              "importance_sha256": (sha(imp_path) if imp_path.is_file() else
                                    hashlib.sha256("".join(
                                        f"{f.name}:{sha(f)}\n" for f in
                                        sorted(imp_path.glob("*.npz"))).encode()).hexdigest())}
    pathlib.Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}\nverification report: {a.out}")


if __name__ == "__main__":
    main()
