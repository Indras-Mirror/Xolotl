#!/usr/bin/env python3
"""Apply the actual abliteration LoRA to an ordinary Q8_0 GGUF, in place.

The source adapter stores W_new = W_base + scale * B @ A.  Rank 1 tensors are
recomputed against the dequantized Q8 trunk (the documented re-export); rank-k
tensors use their actual B @ A, not a rank-1 approximation.  The Q8 file is
both the calibration input and the source for the later quantize pass, so the
edit necessarily precedes imatrix, dumps, screening and allocation.

Use --prepare FIRST to record a rollback.  There are two honest kinds:

  --range-backup DIR   a tensor_backup.py range backup of exactly the tensors
                       this edit rewrites.  Complete for the edit, ~1/4 the
                       bytes of a whole-file copy, and verified against the
                       model at prepare time.  This is the supported path.
  --pristine FILE      an independent full copy on another file.

The old --backup hard link is not a rollback for an in-place edit: it shares
the inode, so writing at data_offset changes both names.  It is kept only for
a model that is a separate file from the one being edited.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V41 = HERE.parent / "v41"
sys.path.insert(0, str(V41))
import v41_formats as F  # noqa: E402

sys.path.insert(0, "/home/mal/AI/llama.cpp-master-rebase/gguf-py")
import gguf  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def pairs(adapter):
    out = {}
    for t in gguf.GGUFReader(adapter).tensors:
        base, kind = t.name.rsplit(".lora_", 1)
        # gguf-py reverses the GGUF dimensions for numpy. Dense factors are
        # [rank,out] / [in,rank]; expert factors are [expert,out,rank] /
        # [expert,rank,in]. Normalize only the dense pair to out-major form.
        value = np.asarray(t.data, dtype=np.float32)
        out.setdefault(base, {})[kind] = value.T if value.ndim == 2 else value
    return out


def q8_rows(t, first=0, count=None):
    """Q8_0 rows in numpy order; no transposition of the GGUF ne0 axis."""
    if t.tensor_type != gguf.GGMLQuantizationType.Q8_0:
        raise ValueError(f"{t.name}: {t.tensor_type.name}, expected Q8_0")
    n_in, nb = int(t.shape[0]), int(t.shape[0]) // 32
    assert n_in % 32 == 0
    count = int(t.data.size // t.data.shape[-1]) - first if count is None else count
    raw = np.asarray(t.data).reshape(-1, nb, 34)[first:first + count]
    d = raw[:, :, :2].copy().reshape(-1).view(np.float16).astype(np.float32)
    d = d.reshape(count, nb, 1)
    q = raw[:, :, 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(tuple(t.data.shape[:-1]) + (n_in,))


def apply_pair(W, B, A, scale, recompute_rank1):
    """Apply one dense matrix or a chunk of expert matrices."""
    W = np.asarray(W)
    original_shape = W.shape
    dense = W.ndim == 2
    if dense:
        W = W[None, ...]
    if W.ndim == 4 and W.shape[0] == 1:
        W = W[0]
        original_shape = W.shape
    if W.ndim != 3:
        raise ValueError(f"unsupported weight shape {W.shape}")
    E, out, inn = W.shape
    if B.ndim == 2:
        B, A = B.T, A.T
    if B.ndim == 2:
        B, A = B[None, ...], A[None, ...]
    if B.shape[0] == 1 and E > 1:
        B = np.broadcast_to(B, (E,) + B.shape[1:])
        A = np.broadcast_to(A, (E,) + A.shape[1:])
    if B.ndim != 3 or A.ndim != 3:
        raise ValueError(f"unsupported factor shapes B={B.shape} A={A.shape}")
    if B.shape[0] != E or B.shape[1] != out or A.shape[0] != E or A.shape[2] != inn or B.shape[2] != A.shape[1]:
        raise ValueError(f"edit shape mismatch W={W.shape} B={B.shape} A={A.shape}")
    rank = A.shape[1]
    if rank == 1 and recompute_rank1:
        norms = np.linalg.norm(B[:, :, 0], axis=1)
        if np.any(norms == 0):
            raise ValueError("zero B direction")
        u = B[:, :, 0] / norms[:, None]
        A = -(scale / norms)[:, None, None] * np.einsum("eo,eoi->ei", u, W)[:, None, :]
    else:
        A = A * scale
    return (W + np.einsum("eor,eri->eoi", B, A)).reshape(original_shape).astype(np.float32)


def encode_q8(W):
    if W.ndim == 2:
        return F.encode_q8_0(W)
    return np.stack([F.encode_q8_0(row) for row in W])


def decode_q8(raw, n_in, shape):
    return np.stack([F.decode(F.Q8_0, row, n_in)
                     for row in raw.reshape(-1, raw.shape[-1])]).reshape(shape)


def edit_survival(base, edited, encoded):
    """Return alpha_eff and signal; Q(W)-W noise is NOT counted as edit signal."""
    intended = edited - base
    realized = encoded - base
    den = float(np.sum(intended.astype(np.float64) ** 2))
    num = float(np.sum(realized.astype(np.float64) * intended.astype(np.float64)))
    alpha = num / den if den else 1.0
    signal = 100 * num * num / (den * float(np.sum(realized.astype(np.float64) ** 2))) if den and np.any(realized) else 0.0
    rel = float(np.linalg.norm(encoded - edited) / (np.linalg.norm(base) or 1.0))
    return {"alpha_eff": alpha, "signal_pct": signal, "requant_error_rel": rel}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_range(f, offset, nbytes):
    f.seek(offset)
    out = bytearray()
    while len(out) < nbytes:
        b = f.read(min(nbytes - len(out), 16 << 20))
        if not b:
            raise IOError(f"short read at {offset}+{len(out)}")
        out += b
    return bytes(out)


def load_range_backup(d, model, names):
    """Validate a tensor_backup.py backup against the model, return (blob, index).

    Every tensor in `names` must be covered, and the bytes it covers must still
    be the model's current bytes: that is what makes the backup a rollback for
    an edit that has not happened yet.
    """
    d = pathlib.Path(d)
    man_path = d / "backup.json"
    if not man_path.is_file():
        raise SystemExit(f"{d}: not a finished backup (no backup.json)")
    man = json.loads(man_path.read_text())
    if pathlib.Path(man["model"]).resolve() != pathlib.Path(model).resolve():
        raise SystemExit(f"backup is for {man['model']}, not {model}")
    blob = pathlib.Path(man["blob"])
    if not blob.is_file():
        raise SystemExit(f"backup blob missing: {blob}")
    by = {t["name"]: t for t in man["tensors"]}
    missing = [n for n in names if n not in by]
    if missing:
        raise SystemExit(f"backup does not cover {len(missing)} target(s), e.g. {missing[:5]}")
    size = os.path.getsize(model)
    for n in names:
        r = by[n]
        if r["offset"] + r["nbytes"] > size:
            raise SystemExit(f"{n}: backup range extends past {model}")
    return blob, {n: by[n] for n in names}


class _BackupTensor:
    """A tensor stand-in whose payload comes from a range backup.

    It carries the live tensor's shape/type so `q8_rows` decodes it exactly as
    it decodes the model, without re-deriving gguf-py's byte layout here.
    """

    def __init__(self, t, raw):
        if len(raw) != int(t.n_bytes):
            raise ValueError(f"{t.name}: backup has {len(raw)} bytes, tensor needs {t.n_bytes}")
        self.name = t.name
        self.shape = t.shape
        self.tensor_type = t.tensor_type
        self.data_offset = t.data_offset
        self.data = np.frombuffer(raw, dtype=np.uint8).reshape(t.data.shape)


def check_targets_unchanged(model, state, tensors, names):
    """Refuse an edit when any target no longer matches its recorded backup.

    A second application of the same adapter is not idempotent, so this runs
    before the first write whenever there is no journal to resume.
    """
    with open(model, "rb") as f:
        for n in sorted(names):
            t = tensors[n]
            want = state["tensor_sha256"][n]
            if sha_bytes(read_range(f, int(t.data_offset), int(t.n_bytes))) != want:
                raise SystemExit(f"{n} differs from the backup; refusing a double edit")


def prepare(model, backup, report, pristine=None, pristine_sha=None, range_backup=None, names=None):
    """Record a rollback for an in-place edit.

    --backup hard-links `model` on the same filesystem. That is only a rollback
    if the edit target is a SEPARATE file: a hard link shares the inode, so an
    in-place write at data_offset changes both names. `--pristine` instead points
    at an existing independent full copy and proves it is one.
    """
    model = pathlib.Path(model)
    if range_backup:
        if not names:
            raise SystemExit("a range backup needs the target tensor list")
        blob, idx = load_range_backup(range_backup, model, names)
        print(f"verifying {len(names)} covered ranges against the model "
              f"({sum(r['nbytes'] for r in idx.values()) / 2**30:.3f} GiB)...", flush=True)
        with open(model, "rb") as f, open(blob, "rb") as b:
            for n, r in sorted(idx.items()):
                if sha_bytes(read_range(b, r["blob_offset"], r["nbytes"])) != r["sha256"]:
                    raise SystemExit(f"{n}: backup blob is corrupt")
                if read_range(f, r["offset"], r["nbytes"]) != read_range(b, r["blob_offset"], r["nbytes"]):
                    raise SystemExit(f"{n}: model already differs from the backup; refuse to edit")
        info = {"schema": 1, "state": "prepared", "model": str(model),
                "backup": str(pathlib.Path(range_backup).resolve()),
                "rollback_kind": "range-backup",
                "source_size": model.stat().st_size,
                "tensor_sha256": {n: r["sha256"] for n, r in idx.items()},
                "created": time.strftime("%F %T")}
        pathlib.Path(report).write_text(json.dumps(info, indent=1) + "\n")
        print(f"rollback verified (range-backup): {info['backup']}\n"
              f"{len(idx)} tensors, {sum(r['nbytes'] for r in idx.values()) / 2**30:.3f} GiB covered")
        return
    if pristine:
        pristine = pathlib.Path(pristine)
        if not pristine.is_file():
            raise SystemExit(f"pristine copy missing: {pristine}")
        if pristine.resolve() == model.resolve():
            raise SystemExit("--pristine is the edit target itself")
        a, b = model.stat(), pristine.stat()
        if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
            raise SystemExit("--pristine is a hard link to the edit target; that is not a copy")
        if a.st_size != b.st_size:
            raise SystemExit(f"pristine size {b.st_size} != model size {a.st_size}")
        sha = pristine_sha or digest(pristine)
        got = digest(model)
        if got != sha:
            raise SystemExit(f"model SHA-256 {got} != pristine SHA-256 {sha}")
        info = {"schema": 1, "state": "prepared", "model": str(model),
                "backup": str(pristine), "rollback_kind": "independent-copy",
                "source_size": a.st_size, "source_sha256": sha,
                "created": time.strftime("%F %T")}
    else:
        backup = pathlib.Path(backup)
        if backup.exists():
            raise SystemExit(f"refusing to overwrite existing rollback {backup}")
        if model.stat().st_dev != backup.parent.stat().st_dev:
            raise SystemExit("rollback must be on the SAME filesystem for a hard link")
        os.link(model, backup)
        if os.stat(model).st_ino != os.stat(backup).st_ino:
            raise SystemExit("rollback is not a hard link")
        info = {"schema": 1, "state": "prepared", "model": str(model), "backup": str(backup),
                "rollback_kind": "hardlink-same-inode",
                "source_size": model.stat().st_size, "source_sha256": digest(model),
                "created": time.strftime("%F %T")}
    pathlib.Path(report).write_text(json.dumps(info, indent=1) + "\n")
    print(f"rollback verified ({info['rollback_kind']}): {info['backup']}\n"
          f"source SHA-256: {info['source_sha256']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    ap.add_argument("--adapter", default="/media/mal/NVME1TB/Models/Qwen3.8-Flash-Next-Abliterated-Adapter/Q5_K_M/adapter.gguf")
    ap.add_argument("--backup", help="hard link on the same filesystem (only safe if --model is a separate file)")
    ap.add_argument("--pristine", help="independent full copy used as the rollback for an in-place edit")
    ap.add_argument("--pristine-sha256", help="skip re-hashing the pristine copy; must match --model")
    ap.add_argument("--range-backup", help="tensor_backup.py directory covering exactly this edit's tensors")
    ap.add_argument("--report", default="/home/mal/.quetza-data/qwen38-q8-edit.json")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--layers", default="", help="comma list; default all adapter layers (testing only)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.model:
        ap.error("--model is required")
    r = gguf.GGUFReader(a.model)
    tensors = {t.name: t for t in r.tensors}
    edit = pairs(a.adapter)
    layers = {int(x) for x in a.layers.split(",") if x.strip()} if a.layers else None
    selected = {n: ab for n, ab in edit.items()
                if layers is None or int(n.split(".")[1]) in layers}
    missing = sorted(set(selected) - set(tensors))
    if missing:
        raise SystemExit(f"model lacks adapter targets: {missing[:5]}")
    for n, t in tensors.items():
        if n not in selected:
            continue
        if t.tensor_type != gguf.GGMLQuantizationType.Q8_0:
            raise SystemExit(f"{n}: expected Q8_0, got {t.tensor_type.name}")
        if not np.allclose(t.data_offset % 4, 0):
            raise SystemExit(f"{n}: unaligned data offset")
    if a.prepare:
        kinds = [k for k, v in (("--range-backup", a.range_backup), ("--pristine", a.pristine),
                                ("--backup", a.backup)) if v]
        if len(kinds) != 1:
            ap.error(f"give exactly one rollback: {', '.join(kinds) or 'none given'}")
        prepare(a.model, a.backup, a.report, a.pristine, a.pristine_sha256,
                a.range_backup, sorted(selected))
        return
    if a.pristine or a.range_backup:
        ap.error("--pristine/--range-backup are only used with --prepare")
    if not a.dry_run:
        report = pathlib.Path(a.report)
        if not report.is_file():
            raise SystemExit("no preparation report; run --prepare first")
        state = json.loads(report.read_text())
        if state.get("model") != str(pathlib.Path(a.model)):
            raise SystemExit("report belongs to another model")
        kind = state.get("rollback_kind")
        if kind == "range-backup":
            if not pathlib.Path(state["backup"], "backup.json").is_file():
                raise SystemExit("range backup is missing")
        elif not pathlib.Path(state["backup"]).is_file():
            raise SystemExit("pristine rollback is missing")
        journal = pathlib.Path(str(report) + ".journal.json")
        completed = {}
        if journal.exists():
            previous = json.loads(journal.read_text())
            if previous.get("rollback") != state["backup"]:
                raise SystemExit("edit journal is for a different rollback")
            if previous.get("adapter") != a.adapter or previous.get("scale") != a.scale:
                raise SystemExit("edit journal is for a different adapter or scale")
            completed = {r["tensor"]: r for r in previous.get("targets", [])}
        elif kind == "range-backup":
            print("checking every target still matches its backup...", flush=True)
            check_targets_unchanged(a.model, state, tensors, selected)
        elif digest(a.model) != state["source_sha256"]:
            raise SystemExit("model differs from the verified pristine source; refusing a double edit")
    else:
        completed, kind = {}, None
    rows = []
    for i, (name, ab) in enumerate(sorted(selected.items()), 1):
        t = tensors[name]
        B, A = ab["b"], ab["a"]
        rank = A.shape[1] if A.ndim == 3 else A.shape[0]
        if name in completed:
            old = completed[name]
            if old.get("rank") != rank:
                raise SystemExit(f"{name}: adapter rank changed since the journal")
            W = q8_rows(t)
            # Recompute the exact output from the pristine rollback to validate
            # the saved payload, not merely a same-sized tensor region.
            if kind == "range-backup":
                bdir = pathlib.Path(state["backup"])
                rec = {x["name"]: x for x in json.loads((bdir / "backup.json").read_text())["tensors"]}[name]
                with open(bdir / "tensors.bin", "rb") as bf:
                    base = q8_rows(_BackupTensor(t, read_range(bf, rec["blob_offset"], rec["nbytes"])))
            else:
                pristine = gguf.GGUFReader(state["backup"])
                pt = next(x for x in pristine.tensors if x.name == name)
                base = q8_rows(pt)
                del pristine, pt
            expected = apply_pair(base, B, A, a.scale, recompute_rank1=(rank == 1))
            raw = encode_q8(expected)
            del base, expected
            with open(a.model, "rb") as f:
                f.seek(int(t.data_offset))
                if f.read(raw.nbytes) != raw.tobytes():
                    raise SystemExit(f"{name}: journal payload differs; restore the pristine backup before retrying")
            rows.append(old)
            print(f"[{i}/{len(selected)}] {name}: verified in journal", flush=True)
            continue
        W = q8_rows(t)
        before = W.copy()
        edited = apply_pair(W, B, A, a.scale, recompute_rank1=(rank == 1))
        if not np.all(np.isfinite(edited)):
            raise ValueError(f"{name}: nonfinite edited weights")
        raw = encode_q8(edited)
        got = decode_q8(raw, int(t.shape[0]), edited.shape)
        metric = edit_survival(before, edited, got)
        rec = {"tensor": name, "rank": rank, "n_rows": int(W.shape[0]),
               "edit_rel": float(np.linalg.norm(edited - before) / (np.linalg.norm(before) or 1.0)),
               **metric}
        rows.append(rec)
        print(f"[{i}/{len(selected)}] {name}: rank={rank} edit={rec['edit_rel']:.4%} "
              f"alpha_eff={metric['alpha_eff']:.5f} signal={metric['signal_pct']:.2f}%", flush=True)
        if not a.dry_run:
            with open(a.model, "r+b", buffering=0) as f:
                f.seek(int(t.data_offset))
                f.write(raw.tobytes())
            # Byte-level payload readback; do not declare an in-place write complete from write() alone.
            with open(a.model, "rb") as f:
                f.seek(int(t.data_offset))
                if f.read(raw.nbytes) != raw.tobytes():
                    raise IOError(f"{name}: payload readback differs")
            if not a.dry_run:
                journal_state = {"rollback": state["backup"],
                                 "adapter": a.adapter, "scale": a.scale, "targets": rows}
                tmp = pathlib.Path(str(report) + ".journal.json.tmp")
                tmp.write_text(json.dumps(journal_state, indent=1) + "\n")
                os.replace(tmp, pathlib.Path(str(report) + ".journal.json"))
    result = {"schema": 1, "state": "dry-run" if a.dry_run else "edited",
              "model": a.model, "adapter": a.adapter, "adapter_sha256": digest(a.adapter),
              "scale": a.scale, "layers": sorted(layers) if layers is not None else "all",
              "targets": rows}
    if not a.dry_run:
        result["edited_sha256"] = digest(a.model)
    pathlib.Path(a.report).write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {a.report}: {len(rows)} target tensors; state={result['state']}")


if __name__ == "__main__":
    main()
